"""Persistent-delegate tool: invoke a named agent as a subagent.

This is the H3 piece of the ``feat/persistent-agents-hipp0`` stack.
It's the orchestration glue between:

  - :mod:`hermes_cli.agent_registry` — loads the named agent's
    SOUL.md + config.yaml off disk.
  - :class:`agent.hipp0_memory_provider.Hipp0MemoryProvider` — runs
    the HIPP0 session lifecycle (start → compile → capture → end),
    WAL + degraded mode included.
  - A Hermes ``AIAgent`` subagent (or any injectable runner) that
    actually drives the LLM conversation under the delegate's SOUL.

Flow for a single :meth:`PersistentDelegateTool.invoke` call::

    1. AgentRegistry.get_agent(name)        -> AgentProfile
    2. Hipp0MemoryProvider(agent + hipp0)   -> bound provider
    3. provider.start_session(...)          -> server UUID
    4. provider.compile(task, fast=True)    -> CompiledContext
    5. build system prompt = SOUL + compiled context + degraded warn
    6. runner(task, system_prompt)          -> delegate's final reply
    7. provider.capture(transcript)         -> 202 (async snippet extract)
    8. return {response, session_id, captured, degraded, ...}

:func:`persistent_delegate_task_handler` is the thin sync wrapper
registered with ``tools.registry`` so the model can fire off
``persistent_delegate_task`` from a parent agent. It runs the async
flow on a fresh event loop (or inside the parent's loop when one is
active) and forwards the result through the standard tool-reply
protocol.

The runner is injected so tests can exercise the full orchestration
without spinning up a real LLM — see
``tests/tools/test_persistent_delegate_tool.py``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional

import httpx

from agent.hipp0_memory_provider import (
    CompiledContext,
    Hipp0MemoryProvider,
    Hipp0UnavailableError,
)
from hermes_cli.agent_registry import (
    AgentNotFoundError,
    AgentProfile,
    get_agent,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


# A runner takes (task_description, system_prompt, context) and returns
# the delegate's final message + a raw transcript (string form suitable
# for HIPP0 capture). Kept narrow so tests can inject a trivial stub
# and production code can wrap AIAgent.run_conversation.
DelegateRunner = Callable[
    [str, str, "DelegateRunContext"],
    Awaitable["DelegateRunResult"],
]


@dataclass
class DelegateRunContext:
    """Everything the runner needs beyond ``task`` and ``system_prompt``."""

    agent: AgentProfile
    session_id: str
    compiled: CompiledContext
    provider: Hipp0MemoryProvider
    parent_agent: Optional[Any] = None  # parent AIAgent, when available


@dataclass
class DelegateRunResult:
    """Result returned by a :data:`DelegateRunner`."""

    final_message: str
    transcript: str  # full text to hand to HIPP0 capture


@dataclass
class PersistentDelegateResult:
    """Public shape of an :meth:`PersistentDelegateTool.invoke` call."""

    agent_name: str
    session_id: str
    response: str
    captured: Dict[str, Any]
    compiled_degraded: bool
    compiled_token_count: int
    degraded_reason: Optional[str] = None

    def to_tool_payload(self) -> Dict[str, Any]:
        """Serializable dict for the outer `handle_function_call` result."""
        return {
            "agent_name": self.agent_name,
            "session_id": self.session_id,
            "response": self.response,
            "captured": self.captured,
            "compiled_degraded": self.compiled_degraded,
            "compiled_token_count": self.compiled_token_count,
            "degraded_reason": self.degraded_reason,
        }


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class PersistentDelegateError(RuntimeError):
    """Raised when a persistent delegate cannot complete its task."""


class PersistentDelegateConfigError(PersistentDelegateError):
    """Raised when HIPP0 env / agent config is insufficient to run a delegate."""


# ---------------------------------------------------------------------------
# Core tool
# ---------------------------------------------------------------------------


class PersistentDelegateTool:
    """Orchestrates a persistent named-agent delegation.

    Parameters
    ----------
    base_url, api_key:
        HIPP0 endpoint and bearer token. Defaults pull from
        ``HIPP0_BASE_URL`` / ``HIPP0_API_KEY`` env vars.
    runner:
        Injectable coroutine that actually runs the delegate
        conversation. The default runner wraps Hermes's ``AIAgent``
        via :func:`_default_aiagent_runner`; tests pass a stub.
    provider_factory:
        Injectable factory that builds the :class:`Hipp0MemoryProvider`.
        Default wires the real provider. Tests inject a factory that
        uses a shared httpx.AsyncClient pointed at the mock HIPP0.
    """

    def __init__(
        self,
        *,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        runner: Optional[DelegateRunner] = None,
        provider_factory: Optional[
            Callable[[AgentProfile], Hipp0MemoryProvider]
        ] = None,
    ) -> None:
        self.base_url = base_url or os.environ.get(
            "HIPP0_BASE_URL", "http://localhost:3000"
        )
        self.api_key = api_key or os.environ.get("HIPP0_API_KEY", "")
        self._runner = runner or _default_aiagent_runner
        self._provider_factory = provider_factory

    async def invoke(
        self,
        agent_name: str,
        task: str,
        *,
        platform: str = "cli",
        user_id: Optional[str] = None,
        external_chat_id: Optional[str] = None,
        parent_agent: Optional[Any] = None,
        end_session: bool = False,
    ) -> PersistentDelegateResult:
        """Run a persistent delegate end-to-end.

        Parameters
        ----------
        agent_name:
            Name registered under ``<hermes_root>/agents/<name>/``.
        task:
            The user-visible task description.
        platform:
            Platform the delegation is happening on (``"cli"``,
            ``"telegram"``, …). Forwarded to HIPP0 session/start.
        user_id, external_chat_id:
            Optional platform identifiers. Passed to HIPP0 so sessions
            can be scoped per-user.
        parent_agent:
            Parent ``AIAgent`` instance, if one is driving the
            delegation. Exposed to the runner via ``DelegateRunContext``
            so implementations can inherit credentials, display, etc.
        end_session:
            When True, call ``session/end`` after a successful capture.
            Telegram keeps the session open across messages; one-shot
            CLI invocations should pass True so HIPP0 runs the
            rolling-summary flush.
        """
        try:
            profile = get_agent(agent_name)
        except AgentNotFoundError as e:
            raise PersistentDelegateError(
                f"Persistent delegate {agent_name!r} not registered: {e}"
            ) from e

        if not profile.config.project_id:
            raise PersistentDelegateConfigError(
                f"Agent {agent_name!r} has no project_id. Register with HIPP0 "
                f"via Hipp0MemoryProvider.register() and persist the result "
                f"via hermes_cli.agent_registry.update_agent_config."
            )
        if not self.api_key:
            raise PersistentDelegateConfigError(
                "HIPP0_API_KEY env var is not set. Cannot invoke a "
                "persistent delegate without HIPP0 credentials."
            )

        provider = self._make_provider(profile)
        try:
            session_id = await provider.start_session(
                platform=platform,
                user_id=user_id,
                external_chat_id=external_chat_id,
            )
            compiled = await provider.compile(
                task_description=task,
                fast_mode=True,
            )
            system_prompt = self._build_system_prompt(
                profile, compiled, platform=platform,
            )

            run_ctx = DelegateRunContext(
                agent=profile,
                session_id=session_id,
                compiled=compiled,
                provider=provider,
                parent_agent=parent_agent,
            )
            run_result = await self._runner(task, system_prompt, run_ctx)
            if not isinstance(run_result, DelegateRunResult):
                raise PersistentDelegateError(
                    f"Delegate runner returned {type(run_result).__name__}, "
                    "expected DelegateRunResult"
                )

            try:
                captured = await provider.capture(
                    run_result.transcript,
                    source="hermes",
                    source_channel=external_chat_id,
                )
            except Hipp0UnavailableError as e:
                # Capture failed after retries — the WAL already has it.
                # Don't fail the delegation; log and proceed.
                logger.warning(
                    "persistent delegate %r: capture WALed (%s)",
                    agent_name,
                    e,
                )
                captured = {"status": "walled", "capture_id": None}

            if end_session:
                try:
                    await provider.end_session()
                except Hipp0UnavailableError as e:
                    logger.warning(
                        "persistent delegate %r: session/end failed: %s",
                        agent_name,
                        e,
                    )

            return PersistentDelegateResult(
                agent_name=agent_name,
                session_id=session_id,
                response=run_result.final_message,
                captured=captured,
                compiled_degraded=compiled.degraded,
                compiled_token_count=compiled.total_tokens,
                degraded_reason=compiled.degraded_reason,
            )
        finally:
            await provider.aclose()

    # ------------------------------------------------------------------ helpers

    def _make_provider(self, profile: AgentProfile) -> Hipp0MemoryProvider:
        if self._provider_factory is not None:
            return self._provider_factory(profile)
        return Hipp0MemoryProvider(
            base_url=self.base_url,
            api_key=self.api_key,
            project_id=str(profile.config.project_id),
            agent_name=profile.name,
            agent_id=str(profile.config.agent_id or ""),
            pending_wal_path=profile.pending_wal_path,
            memory_md_path=profile.memory_path,
        )

    @staticmethod
    def _build_system_prompt(
        profile: AgentProfile,
        compiled: CompiledContext,
        *,
        platform: Optional[str] = None,
    ) -> str:
        """Compose the delegate's system prompt via the slim builder.

        Delegates don't want the full Hermes operating manual — their
        persona (SOUL.md) is authoritative and HIPP0 owns their memory.
        Delegates always use :func:`agent.prompt_builder.build_slim_system_prompt`.
        Degraded-mode annotations come from :meth:`CompiledContext.as_prompt_block`.
        """
        from agent.prompt_builder import build_slim_system_prompt

        return build_slim_system_prompt(
            profile.soul,
            compiled_context_block=compiled.as_prompt_block(),
            platform_hint=platform,
        )


# ---------------------------------------------------------------------------
# Default runner — AIAgent-backed (lazy import so test runs don't need the
# full Hermes runtime to exercise this module).
# ---------------------------------------------------------------------------


async def _default_aiagent_runner(
    task: str,
    system_prompt: str,
    ctx: DelegateRunContext,
) -> DelegateRunResult:
    """Default runner: spawn a lightweight AIAgent and drive a single turn.

    Runs synchronously inside an executor because ``AIAgent.chat`` is
    blocking. The delegate inherits model / credentials from the parent
    when one is provided, otherwise falls back to the agent's own
    config.yaml.
    """
    parent = ctx.parent_agent

    def _run_sync() -> DelegateRunResult:
        from run_agent import AIAgent  # lazy import

        agent_cfg = ctx.agent.config
        model = (getattr(parent, "model", None) or agent_cfg.model)
        base_url = getattr(parent, "base_url", None)
        api_key = getattr(parent, "api_key", None)

        delegate = AIAgent(
            model=model,
            base_url=base_url,
            api_key=api_key,
            max_iterations=50,
            quiet_mode=True,
            ephemeral_system_prompt=system_prompt,
            platform=getattr(parent, "platform", "cli"),
            skip_context_files=True,
            skip_memory=True,  # Hipp0 provider owns memory, not builtin
            log_prefix=f"[persistent:{ctx.agent.name}]",
        )
        delegate._delegate_depth = (getattr(parent, "_delegate_depth", 0) or 0) + 1

        final = delegate.chat(task)
        transcript_lines = [
            f"USER: {task}",
            f"ASSISTANT({ctx.agent.name}): {final}",
        ]
        return DelegateRunResult(
            final_message=final,
            transcript="\n".join(transcript_lines),
        )

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _run_sync)


# ---------------------------------------------------------------------------
# Tool registration — sync bridge for tools/registry.py
# ---------------------------------------------------------------------------


PERSISTENT_DELEGATE_TASK_SCHEMA = {
    "name": "persistent_delegate_task",
    "description": (
        "Hand a task to a persistent named agent (e.g. 'alice', 'bob') whose "
        "memory lives in HIPP0. The named agent shares cross-session memory "
        "with every other persistent agent in the same project — anything "
        "Alice learns is retrievable by Bob.\n\n"
        "Use this INSTEAD of delegate_task when:\n"
        "- The task should be handled by a specific persona (sales, product, …).\n"
        "- You want the conversation to feed the shared HIPP0 memory graph.\n"
        "- You want the delegate to remember past user preferences.\n\n"
        "The tool runs the delegate end-to-end: HIPP0 session start, context "
        "compile, delegate conversation, transcript capture, optional "
        "session end. The returned payload includes the delegate's final "
        "message plus a 'captured.capture_id' you can poll with "
        "GET /api/capture/:id if you need extracted snippet ids."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "agent_name": {
                "type": "string",
                "description": (
                    "Name of the persistent agent (e.g. 'alice'). Must exist "
                    "under <hermes_root>/agents/."
                ),
            },
            "task": {
                "type": "string",
                "description": "Task description the delegate should act on.",
            },
            "platform": {
                "type": "string",
                "description": (
                    "Platform the delegation is happening on "
                    "('cli', 'telegram', 'discord', 'web'). Defaults to 'cli'."
                ),
            },
            "user_id": {
                "type": "string",
                "description": "Optional external user id (e.g. Telegram user id).",
            },
            "external_chat_id": {
                "type": "string",
                "description": "Optional external chat id (e.g. Telegram chat id).",
            },
            "end_session": {
                "type": "boolean",
                "description": (
                    "If true, call HIPP0 session/end after the delegate "
                    "completes. Use for one-shot CLI invocations; keep "
                    "false for long-lived Telegram conversations."
                ),
            },
        },
        "required": ["agent_name", "task"],
    },
}


def persistent_delegate_task_handler(
    args: Dict[str, Any],
    *,
    parent_agent: Optional[Any] = None,
    **_: Any,
) -> str:
    """Sync tool-registry handler.

    Runs :meth:`PersistentDelegateTool.invoke` on a fresh event loop.
    Returns a JSON string (the standard tool-reply shape) so the
    parent AIAgent can inject it into the conversation.
    """
    agent_name = args.get("agent_name")
    task = args.get("task")
    if not agent_name or not task:
        return json.dumps(
            {"error": "persistent_delegate_task requires agent_name and task"}
        )

    tool = PersistentDelegateTool()

    async def _run() -> PersistentDelegateResult:
        return await tool.invoke(
            agent_name=agent_name,
            task=task,
            platform=args.get("platform", "cli"),
            user_id=args.get("user_id"),
            external_chat_id=args.get("external_chat_id"),
            parent_agent=parent_agent,
            end_session=bool(args.get("end_session", False)),
        )

    try:
        result = asyncio.run(_run())
    except PersistentDelegateError as e:
        return json.dumps({"error": str(e)})
    except Exception as e:  # pragma: no cover - defensive
        logger.exception("persistent_delegate_task failed")
        return json.dumps({"error": f"internal error: {e}"})

    return json.dumps(result.to_tool_payload())


# Registry wiring — guarded so importing the module in tests that don't
# rely on the full tool registry (e.g. this phase's unit tests) doesn't
# force all of tools/registry.py's import graph to load.
def _register() -> None:  # pragma: no cover - import-time wiring
    try:
        from tools.registry import registry
    except Exception as e:
        logger.debug("persistent_delegate_tool: registry import skipped (%s)", e)
        return
    registry.register(
        name="persistent_delegate_task",
        toolset="delegation",
        schema=PERSISTENT_DELEGATE_TASK_SCHEMA,
        handler=lambda args, **kw: persistent_delegate_task_handler(
            args, parent_agent=kw.get("parent_agent")
        ),
        emoji="🧠",
    )


_register()
