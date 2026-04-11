"""Tests for :class:`tools.persistent_delegate_tool.PersistentDelegateTool`.

These tests exercise the full orchestration flow (load agent → start
HIPP0 session → compile → run delegate → capture → optional end)
against the in-process mock HIPP0 from ``tests/fixtures/mock_hipp0.py``.
A stub runner is injected so no real LLM is invoked.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import List

import pytest

from agent.hipp0_memory_provider import Hipp0MemoryProvider
from hermes_cli import agent_registry
from hermes_cli.agent_registry import (
    AgentConfig,
    register_agent,
    update_agent_config,
)
from tests.fixtures.mock_hipp0 import MockHipp0, start_mock_hipp0
from tools.persistent_delegate_tool import (
    DelegateRunContext,
    DelegateRunResult,
    PersistentDelegateConfigError,
    PersistentDelegateError,
    PersistentDelegateTool,
    persistent_delegate_task_handler,
)

pytestmark = pytest.mark.asyncio


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _seed_alice(project_id: str = "proj-uuid", agent_id: str = "agent-uuid") -> None:
    register_agent(
        "alice",
        soul="# Alice\nYou are Alice, the sales lead.\n",
        config=AgentConfig(
            model="anthropic/claude-opus-4.6",
            platform_access=["cli", "telegram"],
            project_id=project_id,
            agent_id=agent_id,
        ),
    )


def _provider_factory(base_url: str, api_key: str):
    """Build a provider factory that points at the given mock HIPP0 URL."""

    def _factory(profile):
        return Hipp0MemoryProvider(
            base_url=base_url,
            api_key=api_key,
            project_id=str(profile.config.project_id),
            agent_name=profile.name,
            agent_id=str(profile.config.agent_id or ""),
            pending_wal_path=profile.pending_wal_path,
            memory_md_path=profile.memory_path,
        )

    return _factory


async def _stub_runner(task, system_prompt, ctx: DelegateRunContext):
    """Trivial runner: echoes the task, records the system prompt."""
    return DelegateRunResult(
        final_message=f"[alice] got task: {task}",
        transcript=f"SYSTEM:\n{system_prompt}\n\nUSER: {task}\nASSISTANT: done",
    )


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class TestInvokeHappyPath:
    async def test_invokes_session_compile_run_capture(self, tmp_path):
        _seed_alice()
        async with start_mock_hipp0() as hipp0:
            captured_ctx: List[DelegateRunContext] = []

            async def _spy_runner(task, system_prompt, ctx):
                captured_ctx.append(ctx)
                return await _stub_runner(task, system_prompt, ctx)

            tool = PersistentDelegateTool(
                base_url=hipp0.base_url,
                api_key="test-key",
                runner=_spy_runner,
                provider_factory=_provider_factory(hipp0.base_url, "test-key"),
            )

            result = await tool.invoke(
                "alice",
                "draft a Q3 kickoff email",
                platform="telegram",
                user_id="tg-42",
                external_chat_id="chat-7",
            )

            # Session lifecycle happened.
            assert hipp0.last_call("/api/hermes/session/start") is not None
            # Compile was called with fast-mode params.
            compile_call = hipp0.last_call("/api/compile")
            assert compile_call is not None
            assert compile_call.query.get("format") == "json"
            assert compile_call.query.get("include_patterns") == "false"
            # Capture body has the transcript produced by the runner.
            capture_call = hipp0.last_call("/api/capture")
            assert capture_call is not None
            assert "ASSISTANT: done" in capture_call.body["conversation"]
            assert capture_call.body["session_id"] == result.session_id

            # Runner received the compiled context, not None.
            assert captured_ctx, "runner should have been called"
            assert captured_ctx[0].session_id == result.session_id
            assert captured_ctx[0].compiled.decisions, "compiled context passed"

            # Result payload is populated.
            assert result.agent_name == "alice"
            assert "got task" in result.response
            assert result.compiled_degraded is False
            assert result.captured.get("status") == "processing"
            assert result.session_id

    async def test_end_session_flag(self, tmp_path):
        _seed_alice()
        async with start_mock_hipp0() as hipp0:
            tool = PersistentDelegateTool(
                base_url=hipp0.base_url,
                api_key="test-key",
                runner=_stub_runner,
                provider_factory=_provider_factory(hipp0.base_url, "test-key"),
            )
            await tool.invoke(
                "alice",
                "oneshot",
                platform="cli",
                end_session=True,
            )
            assert hipp0.last_call("/api/hermes/session/end") is not None

    async def test_system_prompt_contains_soul_and_compiled(self, tmp_path):
        _seed_alice()
        async with start_mock_hipp0() as hipp0:
            seen_prompts: List[str] = []

            async def _capture_runner(task, system_prompt, ctx):
                seen_prompts.append(system_prompt)
                return await _stub_runner(task, system_prompt, ctx)

            tool = PersistentDelegateTool(
                base_url=hipp0.base_url,
                api_key="test-key",
                runner=_capture_runner,
                provider_factory=_provider_factory(hipp0.base_url, "test-key"),
            )
            await tool.invoke("alice", "ping")
            assert len(seen_prompts) == 1
            prompt = seen_prompts[0]
            assert "# Alice" in prompt
            assert "sales lead" in prompt
            assert "## Compiled context" in prompt
            assert "Test decision from mock HIPP0" in prompt  # mock default
            assert "DEGRADED" not in prompt


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


class TestInvokeErrors:
    async def test_unknown_agent(self, tmp_path):
        tool = PersistentDelegateTool(
            base_url="http://unused",
            api_key="test-key",
            runner=_stub_runner,
        )
        with pytest.raises(PersistentDelegateError):
            await tool.invoke("ghost", "x")

    async def test_missing_project_id_is_config_error(self, tmp_path):
        register_agent(
            "alice",
            soul="# Alice",
            config=AgentConfig(),  # no project_id
        )
        tool = PersistentDelegateTool(
            base_url="http://unused",
            api_key="test-key",
            runner=_stub_runner,
        )
        with pytest.raises(PersistentDelegateConfigError):
            await tool.invoke("alice", "x")

    async def test_missing_api_key_is_config_error(self, tmp_path, monkeypatch):
        _seed_alice()
        monkeypatch.delenv("HIPP0_API_KEY", raising=False)
        tool = PersistentDelegateTool(
            base_url="http://unused",
            api_key="",
            runner=_stub_runner,
        )
        with pytest.raises(PersistentDelegateConfigError):
            await tool.invoke("alice", "x")

    async def test_runner_returns_wrong_type(self, tmp_path):
        _seed_alice()
        async with start_mock_hipp0() as hipp0:
            async def _bad_runner(task, system_prompt, ctx):
                return "not a DelegateRunResult"  # type: ignore[return-value]

            tool = PersistentDelegateTool(
                base_url=hipp0.base_url,
                api_key="test-key",
                runner=_bad_runner,
                provider_factory=_provider_factory(hipp0.base_url, "test-key"),
            )
            with pytest.raises(PersistentDelegateError):
                await tool.invoke("alice", "x")


# ---------------------------------------------------------------------------
# Degraded compile — runner still runs, system prompt warns
# ---------------------------------------------------------------------------


class TestDegradedCompile:
    async def test_degraded_runs_with_memory_md(self, tmp_path):
        _seed_alice()
        # Write a MEMORY.md for the agent so the degraded path has content.
        mem_path = agent_registry.agents_root() / "alice" / "MEMORY.md"
        mem_path.write_text(
            "# Local snapshot\nUser prefers phone.\n", encoding="utf-8"
        )

        async with start_mock_hipp0() as hipp0:
            hipp0.queue_failure("/api/compile", status=503, count=10)

            seen_prompts: List[str] = []

            async def _capture_runner(task, system_prompt, ctx):
                seen_prompts.append(system_prompt)
                return DelegateRunResult(
                    final_message="ok",
                    transcript=f"USER: {task}\nASSISTANT: ok",
                )

            tool = PersistentDelegateTool(
                base_url=hipp0.base_url,
                api_key="test-key",
                runner=_capture_runner,
                provider_factory=_provider_factory(hipp0.base_url, "test-key"),
            )
            result = await tool.invoke("alice", "help")

            assert result.compiled_degraded is True
            assert result.degraded_reason
            assert "DEGRADED" in seen_prompts[0]
            assert "prefers phone" in seen_prompts[0]
            # Capture should still have gone through successfully.
            assert result.captured.get("status") == "processing"


# ---------------------------------------------------------------------------
# Sync tool-registry handler
# ---------------------------------------------------------------------------


class TestSyncHandler:
    async def test_handler_end_to_end_via_asyncio_run(
        self, tmp_path, monkeypatch
    ):
        _seed_alice()
        async with start_mock_hipp0() as hipp0:
            monkeypatch.setenv("HIPP0_BASE_URL", hipp0.base_url)
            monkeypatch.setenv("HIPP0_API_KEY", "test-key")

            # Patch the tool class to use our stub runner + factory.
            import tools.persistent_delegate_tool as pdt

            real_init = pdt.PersistentDelegateTool.__init__

            def _patched_init(self, *args, **kwargs):
                real_init(
                    self,
                    *args,
                    runner=_stub_runner,
                    provider_factory=_provider_factory(
                        hipp0.base_url, "test-key"
                    ),
                    **kwargs,
                )

            monkeypatch.setattr(
                pdt.PersistentDelegateTool, "__init__", _patched_init
            )

            # persistent_delegate_task_handler calls asyncio.run() internally,
            # which conflicts with the running loop. We need to call it from
            # a thread so it can create its own loop.
            loop = asyncio.get_event_loop()
            result_str = await loop.run_in_executor(
                None,
                lambda: persistent_delegate_task_handler(
                    {"agent_name": "alice", "task": "say hi"},
                ),
            )
            payload = json.loads(result_str)
            assert "error" not in payload
            assert payload["agent_name"] == "alice"
            assert "got task" in payload["response"]

    async def test_handler_missing_args(self, tmp_path):
        result_str = persistent_delegate_task_handler({"agent_name": "alice"})
        payload = json.loads(result_str)
        assert "error" in payload

    async def test_handler_unknown_agent(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HIPP0_API_KEY", "test-key")
        loop = asyncio.get_event_loop()
        result_str = await loop.run_in_executor(
            None,
            lambda: persistent_delegate_task_handler(
                {"agent_name": "ghost", "task": "x"},
            ),
        )
        payload = json.loads(result_str)
        assert "error" in payload
        assert "ghost" in payload["error"]
