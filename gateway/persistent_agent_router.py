"""Persistent-agent routing for gateway adapters.

The gateway normally routes every incoming message to the single
AIAgent instance backing a chat session. The persistent-agent layer
(``feat/persistent-agents-hipp0``) adds a second routing path on top
of that:

  - When a user message contains ``@agent_name`` and ``agent_name`` is
    a registered persistent agent, the message is dispatched to that
    agent's :class:`tools.persistent_delegate_tool.PersistentDelegateTool`
    — NOT to the chat's default AIAgent.
  - A ``/agent <name>`` slash command sets a *sticky* persistent agent
    for the chat: until the user runs ``/agent off`` (or mentions a
    different agent, or the chat is idle for 30 minutes), every
    subsequent plain text message goes to that agent.
  - When no @mention and no sticky agent is set, the router decides
    ``None`` and the adapter's normal flow proceeds untouched.

This module is deliberately platform-independent. The Telegram
adapter is the primary consumer today (wired in
``gateway/platforms/telegram.py``) but any BasePlatformAdapter can
call :meth:`PersistentAgentRouter.route` from its message handler.

Session continuity: one ``(chat_id, agent_name)`` pair keeps the
same HIPP0 session_id until idle for more than the configured
timeout, at which point the next call reopens a new session.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from hermes_cli.agent_registry import (
    AgentNotFoundError,
    agent_exists,
    get_agent,
)
from tools.persistent_delegate_tool import (
    PersistentDelegateError,
    PersistentDelegateResult,
    PersistentDelegateTool,
)

logger = logging.getLogger(__name__)


# Matches ``@agent_name`` where agent_name is the persistent-agent regex
# (lower-case start letter). A message may contain many @mentions; the
# first one that resolves to a registered agent wins.
_MENTION_RE = re.compile(r"(?<![\w@/])@([a-z][a-z0-9_-]{0,63})\b")

# ``/agent <name>`` slash command — sets sticky agent for the chat.
_SLASH_AGENT_RE = re.compile(r"^\s*/agent(?:@\S+)?(?:\s+(\S+))?\s*$")

# How long a chat session stays warm before we reset the HIPP0 session id.
_DEFAULT_IDLE_TIMEOUT_SECONDS = 30 * 60  # 30 minutes

# Reserved agent names — cannot be used as sticky targets.
_RESERVED_AGENT_NAMES = frozenset({"off", "none", "default", "clear"})


# ---------------------------------------------------------------------------
# Decision types
# ---------------------------------------------------------------------------


@dataclass
class _ChatState:
    """Per-chat routing state kept in-process by :class:`PersistentAgentRouter`."""

    sticky_agent: Optional[str] = None
    session_id_by_agent: Dict[str, str] = field(default_factory=dict)
    last_activity: float = 0.0
    # Most-recent exchange for outcome signal collection
    last_session_id: Optional[str] = None
    last_agent_name: Optional[str] = None
    last_snippet_ids: List[str] = field(default_factory=list)


@dataclass
class RouteDecision:
    """What the router decided about an incoming message.

    Exactly one of ``agent_name`` or ``reply`` is set in practice:

    - ``agent_name`` — dispatch the message to this persistent agent.
      The adapter should call :meth:`PersistentAgentRouter.invoke`.
    - ``reply`` — send this text back to the user directly and stop
      (used for the ``/agent`` command's confirmation/ack messages).
    - Neither set — the router declined. The adapter should proceed
      with its normal (non-persistent) flow.

    ``stripped_text`` is the message text with any leading ``@agent``
    token removed so the delegate sees the actual task.
    """

    agent_name: Optional[str] = None
    reply: Optional[str] = None
    consumed: bool = False
    stripped_text: Optional[str] = None
    new_sticky: bool = False


# ---------------------------------------------------------------------------
# Parser — pure, no I/O
# ---------------------------------------------------------------------------


def parse_mention(text: str) -> Tuple[Optional[str], str]:
    """Extract the first persistent-agent @mention from *text*.

    Returns ``(agent_name, stripped_text)``.

    ``agent_name`` is ``None`` if no @mention is found. ``stripped_text``
    is the original text with the leading @mention removed (only the
    leading one — in-sentence mentions are left in place so the
    delegate sees what the user actually said about them).
    """
    if not text:
        return None, text or ""
    match = _MENTION_RE.search(text)
    if not match:
        return None, text
    name = match.group(1)

    # Only strip the mention if it's at the very start of the text (a
    # pure @alice hello message), or preceded only by whitespace.
    prefix = text[: match.start()].strip()
    suffix = text[match.end() :].lstrip()
    if prefix:
        # Mid-sentence mention — leave the text as-is.
        return name, text
    return name, suffix


def parse_slash_agent_command(text: str) -> Optional[str]:
    """Parse a ``/agent <name>`` slash command.

    Returns:
        - The requested agent name (lowercased) on a valid command.
        - ``""`` for ``/agent off`` / ``/agent none`` / ``/agent clear``
          / ``/agent default`` / bare ``/agent`` — all of which
          clear the sticky agent.
        - ``None`` if the text is not a ``/agent`` command at all.
    """
    if not text:
        return None
    m = _SLASH_AGENT_RE.match(text)
    if not m:
        return None
    raw = (m.group(1) or "").strip().lower()
    if not raw or raw in _RESERVED_AGENT_NAMES:
        return ""
    return raw


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------


class PersistentAgentRouter:
    """Decides whether an incoming message should go to a persistent agent.

    Keeps per-chat sticky-agent + session state in-process. Tests
    inject :class:`tools.persistent_delegate_tool.PersistentDelegateTool`
    via *tool_factory* so they don't need a running HIPP0.
    """

    def __init__(
        self,
        *,
        tool_factory: Optional[Callable[[], PersistentDelegateTool]] = None,
        idle_timeout_seconds: float = _DEFAULT_IDLE_TIMEOUT_SECONDS,
        clock: Optional[Callable[[], float]] = None,
    ) -> None:
        self._tool_factory = tool_factory or PersistentDelegateTool
        self._idle_timeout = idle_timeout_seconds
        self._clock = clock or time.time
        self._state: Dict[str, _ChatState] = {}

    # ----------------------------------------------------------------- state

    def _get_state(self, chat_id: str) -> _ChatState:
        state = self._state.get(chat_id)
        if state is None:
            state = _ChatState()
            self._state[chat_id] = state
        return state

    def _maybe_expire(self, state: _ChatState) -> None:
        if state.last_activity <= 0:
            return
        if self._clock() - state.last_activity > self._idle_timeout:
            logger.debug(
                "persistent_agent_router: idle timeout; clearing session ids"
            )
            state.session_id_by_agent.clear()

    def set_sticky_agent(self, chat_id: str, agent_name: str) -> None:
        state = self._get_state(chat_id)
        state.sticky_agent = agent_name
        state.last_activity = self._clock()

    def clear_sticky_agent(self, chat_id: str) -> Optional[str]:
        state = self._get_state(chat_id)
        prior = state.sticky_agent
        state.sticky_agent = None
        state.session_id_by_agent.clear()
        return prior

    def get_sticky_agent(self, chat_id: str) -> Optional[str]:
        return self._get_state(chat_id).sticky_agent

    def list_known_agents(self) -> List[str]:
        from hermes_cli.agent_registry import list_agents
        return list_agents()

    # -------------------------------------------------------------- decide

    def decide(
        self,
        *,
        chat_id: str,
        text: str,
    ) -> RouteDecision:
        """Decide how to route *text* without invoking the delegate.

        The delegate invocation is split out so adapters can await
        :meth:`invoke` separately — tests can exercise pure decision
        logic without touching HIPP0.
        """
        if text is None:
            return RouteDecision()
        state = self._get_state(chat_id)
        self._maybe_expire(state)

        # /agent slash command
        parsed = parse_slash_agent_command(text)
        if parsed is not None:
            if parsed == "":
                prior = self.clear_sticky_agent(chat_id)
                reply = (
                    f"Sticky agent cleared (was @{prior})."
                    if prior
                    else "No sticky agent was set."
                )
                return RouteDecision(reply=reply, consumed=True)
            # /agent <name> — require the agent to exist
            if not agent_exists(parsed):
                known = ", ".join(self.list_known_agents()) or "(none)"
                return RouteDecision(
                    reply=(
                        f"No persistent agent named @{parsed} is registered.\n"
                        f"Known agents: {known}"
                    ),
                    consumed=True,
                )
            self.set_sticky_agent(chat_id, parsed)
            return RouteDecision(
                reply=f"Sticky agent set to @{parsed}.",
                consumed=True,
                new_sticky=True,
            )

        # @mention routing
        mentioned, stripped = parse_mention(text)
        if mentioned and agent_exists(mentioned):
            # Mentions override the sticky agent for this turn only —
            # we do NOT update sticky_agent, because a one-off mention
            # shouldn't change the sticky target.
            state.last_activity = self._clock()
            return RouteDecision(
                agent_name=mentioned,
                stripped_text=stripped,
            )

        # Sticky agent (only if it still exists)
        if state.sticky_agent and agent_exists(state.sticky_agent):
            state.last_activity = self._clock()
            return RouteDecision(
                agent_name=state.sticky_agent,
                stripped_text=text,
            )

        # No match — fall through to the adapter's default flow.
        return RouteDecision()

    # -------------------------------------------------------------- invoke

    async def invoke(
        self,
        *,
        chat_id: str,
        agent_name: str,
        text: str,
        platform: str,
        user_id: Optional[str] = None,
    ) -> PersistentDelegateResult:
        """Dispatch *text* to the persistent agent.

        The router remembers the returned ``session_id`` per
        ``(chat_id, agent_name)`` so the next invocation reuses it
        until idle timeout or an explicit ``/agent`` change.
        """
        try:
            get_agent(agent_name)
        except AgentNotFoundError as e:
            raise PersistentDelegateError(str(e)) from e

        state = self._get_state(chat_id)
        self._maybe_expire(state)

        tool = self._tool_factory()
        result = await tool.invoke(
            agent_name=agent_name,
            task=text,
            platform=platform,
            user_id=user_id,
            external_chat_id=chat_id,
            end_session=False,  # Telegram conversations are long-lived.
        )

        state.session_id_by_agent[agent_name] = result.session_id
        state.last_activity = self._clock()
        # Stash last-exchange context for outcome signal collection
        state.last_session_id = result.session_id
        state.last_agent_name = agent_name
        snippet_ids: List[str] = []
        captured = getattr(result, "captured", None) or {}
        for key in ("snippet_ids", "summary_snippet_ids", "decision_ids"):
            val = captured.get(key) if isinstance(captured, dict) else None
            if isinstance(val, list):
                snippet_ids.extend(str(s) for s in val)
        state.last_snippet_ids = snippet_ids
        return result

    # ---------------------------------------------------------------- utils

    def get_session_id(
        self, chat_id: str, agent_name: str
    ) -> Optional[str]:
        """Return the cached HIPP0 session id for a chat/agent pair."""
        return self._get_state(chat_id).session_id_by_agent.get(agent_name)

    def get_last_exchange(
        self, chat_id: str
    ) -> Optional[Tuple[str, str, List[str]]]:
        """Return (agent_name, session_id, snippet_ids) for the most recent
        persistent-agent exchange in *chat_id*, or ``None`` if none."""
        state = self._state.get(chat_id)
        if not state or not state.last_session_id or not state.last_agent_name:
            return None
        return (
            state.last_agent_name,
            state.last_session_id,
            list(state.last_snippet_ids),
        )

    async def record_outcome_for_last_exchange(
        self,
        chat_id: str,
        outcome: str,
        *,
        signal_source: str,
        note: Optional[str] = None,
    ) -> bool:
        """Record an outcome for the most-recent exchange in *chat_id*.

        Fire-and-forget friendly: returns True on success, False if there
        is nothing to record. Dual-writes to HIPP0 (via the agent's memory
        provider) and to the local SQLite sessions table.
        """
        last = self.get_last_exchange(chat_id)
        if not last:
            return False
        agent_name, session_id, snippet_ids = last
        # Local SQLite update (best-effort, fire-and-forget semantics)
        try:
            from hermes_state import SessionDB
            db = SessionDB()
            detail = {
                "signal_source": signal_source,
                "snippet_ids": snippet_ids,
                "note": note,
            }
            db.record_outcome(session_id, outcome, signal_source, detail=detail)
        except Exception as exc:  # pragma: no cover - never block the caller
            logger.debug("local outcome write failed: %s", exc)
        # HIPP0 update (only if we have snippet_ids — the provider no-ops
        # otherwise, but we also skip instantiating the provider to save work)
        if snippet_ids:
            try:
                from hermes_cli.agent_registry import get_agent
                profile = get_agent(agent_name)
                tool = self._tool_factory()
                provider = tool._make_provider(profile)
                try:
                    provider._session_id = session_id
                    await provider.record_outcome(
                        snippet_ids,
                        outcome,
                        signal_source=signal_source,
                        note=note,
                    )
                finally:
                    await provider.aclose()
            except Exception as exc:  # pragma: no cover
                logger.debug("hipp0 record_outcome failed: %s", exc)
        return True

    def forget_chat(self, chat_id: str) -> None:
        """Drop all state for *chat_id* (used when a chat ends / resets)."""
        self._state.pop(chat_id, None)
