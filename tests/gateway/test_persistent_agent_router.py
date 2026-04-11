"""Tests for :mod:`gateway.persistent_agent_router`.

Exercises the router's decision logic end-to-end against the H1
AgentRegistry and the H3 PersistentDelegateTool (with a stub runner
pointed at the H2 mock HIPP0 server). Also covers the parser layer
in isolation so regex regressions show up immediately.
"""

from __future__ import annotations

import asyncio
from typing import List

import pytest

from gateway.persistent_agent_router import (
    PersistentAgentRouter,
    RouteDecision,
    parse_mention,
    parse_slash_agent_command,
)
from hermes_cli.agent_registry import AgentConfig, register_agent
from tests.fixtures.mock_hipp0 import start_mock_hipp0
from tools.persistent_delegate_tool import (
    DelegateRunContext,
    DelegateRunResult,
    PersistentDelegateTool,
)


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


class TestParseMention:
    @pytest.mark.parametrize(
        "text,expected_name,expected_stripped",
        [
            ("@alice hello", "alice", "hello"),
            ("  @alice  hello", "alice", "hello"),
            ("@alice", "alice", ""),
            ("tell @alice I said hi", "alice", "tell @alice I said hi"),
            ("no mention here", None, "no mention here"),
            ("", None, ""),
            ("email me@example.com please", None, "email me@example.com please"),
            ("@Alice shouted", None, "@Alice shouted"),  # case-sensitive
            ("@1alice bad", None, "@1alice bad"),  # leading digit invalid
            ("@alice-sales help me", "alice-sales", "help me"),
        ],
    )
    def test_parses(self, text, expected_name, expected_stripped):
        name, stripped = parse_mention(text)
        assert name == expected_name
        assert stripped == expected_stripped


class TestParseSlashAgentCommand:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("/agent alice", "alice"),
            ("/agent  Bob", "bob"),  # whitespace + case
            ("/agent", ""),  # clears
            ("/agent off", ""),
            ("/agent none", ""),
            ("/agent default", ""),
            ("/agent clear", ""),
            ("/agent@hermes_bot alice", "alice"),  # bot-suffix form
            ("hello /agent alice", None),  # not at start
            ("/hello", None),
            ("", None),
        ],
    )
    def test_parses(self, text, expected):
        assert parse_slash_agent_command(text) == expected


# ---------------------------------------------------------------------------
# Router.decide — no delegate invocation
# ---------------------------------------------------------------------------


class TestRouterDecide:
    def _seed(self, *names):
        for n in names:
            register_agent(
                n,
                soul=f"# {n}\nYou are {n}.",
                config=AgentConfig(
                    project_id="proj-uuid",
                    agent_id=f"agent-{n}",
                ),
            )

    def test_mention_routes_to_agent(self, tmp_path):
        self._seed("alice")
        router = PersistentAgentRouter()
        decision = router.decide(chat_id="chat-1", text="@alice draft a plan")
        assert decision.agent_name == "alice"
        assert decision.stripped_text == "draft a plan"
        assert decision.reply is None

    def test_mention_unknown_agent_falls_through(self, tmp_path):
        self._seed("alice")
        router = PersistentAgentRouter()
        decision = router.decide(chat_id="chat-1", text="@ghost hi")
        assert decision.agent_name is None
        assert decision.reply is None
        assert decision.consumed is False

    def test_no_mention_no_sticky_falls_through(self, tmp_path):
        router = PersistentAgentRouter()
        decision = router.decide(chat_id="chat-1", text="hello nobody")
        assert decision.agent_name is None
        assert decision.reply is None

    def test_slash_agent_sets_sticky(self, tmp_path):
        self._seed("alice")
        router = PersistentAgentRouter()
        decision = router.decide(chat_id="chat-1", text="/agent alice")
        assert decision.reply is not None
        assert "Sticky agent set" in decision.reply
        assert decision.new_sticky is True
        assert router.get_sticky_agent("chat-1") == "alice"

        follow = router.decide(chat_id="chat-1", text="draft a plan")
        assert follow.agent_name == "alice"
        assert follow.stripped_text == "draft a plan"

    def test_slash_agent_unknown(self, tmp_path):
        self._seed("alice")
        router = PersistentAgentRouter()
        decision = router.decide(chat_id="chat-1", text="/agent ghost")
        assert decision.consumed is True
        assert decision.reply is not None
        assert "No persistent agent" in decision.reply
        assert router.get_sticky_agent("chat-1") is None

    def test_slash_agent_off_clears_sticky(self, tmp_path):
        self._seed("alice")
        router = PersistentAgentRouter()
        router.decide(chat_id="chat-1", text="/agent alice")
        assert router.get_sticky_agent("chat-1") == "alice"

        decision = router.decide(chat_id="chat-1", text="/agent off")
        assert decision.reply is not None
        assert "cleared" in decision.reply.lower()
        assert router.get_sticky_agent("chat-1") is None

    def test_mention_overrides_sticky_for_one_turn_only(self, tmp_path):
        self._seed("alice", "bob")
        router = PersistentAgentRouter()
        router.decide(chat_id="chat-1", text="/agent alice")  # sticky = alice

        decision = router.decide(chat_id="chat-1", text="@bob quick question")
        assert decision.agent_name == "bob"
        # But sticky is still alice — next plain message routes to alice.
        assert router.get_sticky_agent("chat-1") == "alice"

        follow = router.decide(chat_id="chat-1", text="another thing")
        assert follow.agent_name == "alice"

    def test_idle_timeout_clears_session_ids(self, tmp_path):
        self._seed("alice")
        fake_now = [1000.0]

        def clock():
            return fake_now[0]

        router = PersistentAgentRouter(idle_timeout_seconds=60, clock=clock)
        # Stash a fake session id to observe its expiry.
        state = router._get_state("chat-1")  # noqa: SLF001 - test reach-in
        state.sticky_agent = "alice"
        state.session_id_by_agent["alice"] = "session-123"
        state.last_activity = fake_now[0]

        # Idle past timeout.
        fake_now[0] += 120

        decision = router.decide(chat_id="chat-1", text="anything")
        # Sticky still set (timeout only clears session ids, not sticky).
        assert decision.agent_name == "alice"
        assert "alice" not in router._get_state("chat-1").session_id_by_agent

    def test_forget_chat_resets_all_state(self, tmp_path):
        self._seed("alice")
        router = PersistentAgentRouter()
        router.decide(chat_id="chat-1", text="/agent alice")
        assert router.get_sticky_agent("chat-1") == "alice"
        router.forget_chat("chat-1")
        assert router.get_sticky_agent("chat-1") is None


# ---------------------------------------------------------------------------
# Router.invoke — hits PersistentDelegateTool with stub runner + mock HIPP0
# ---------------------------------------------------------------------------


pytestmark_invoke = pytest.mark.asyncio


class TestRouterInvoke:
    def _seed(self):
        register_agent(
            "alice",
            soul="# Alice\nYou are alice.",
            config=AgentConfig(
                project_id="proj-uuid",
                agent_id="agent-alice",
            ),
        )

    @pytest.mark.asyncio
    async def test_invoke_end_to_end_caches_session_id(self, tmp_path):
        self._seed()
        async with start_mock_hipp0() as hipp0:

            async def _stub_runner(task, system_prompt, ctx: DelegateRunContext):
                return DelegateRunResult(
                    final_message=f"alice says: {task}",
                    transcript=f"USER: {task}\nASSISTANT: alice says: {task}",
                )

            def _tool_factory():
                from agent.hipp0_memory_provider import Hipp0MemoryProvider

                def _pf(profile):
                    return Hipp0MemoryProvider(
                        base_url=hipp0.base_url,
                        api_key="test-key",
                        project_id=str(profile.config.project_id),
                        agent_name=profile.name,
                        agent_id=str(profile.config.agent_id or ""),
                        pending_wal_path=profile.pending_wal_path,
                        memory_md_path=profile.memory_path,
                    )

                return PersistentDelegateTool(
                    base_url=hipp0.base_url,
                    api_key="test-key",
                    runner=_stub_runner,
                    provider_factory=_pf,
                )

            router = PersistentAgentRouter(tool_factory=_tool_factory)

            result = await router.invoke(
                chat_id="chat-1",
                agent_name="alice",
                text="draft a plan",
                platform="telegram",
                user_id="tg-42",
            )
            assert "alice says" in result.response
            assert router.get_session_id("chat-1", "alice") == result.session_id

            # Second invocation reuses state (we don't assert session id
            # equality — the router only cached it, PersistentDelegateTool
            # starts a new session each invoke() for now).
            result2 = await router.invoke(
                chat_id="chat-1",
                agent_name="alice",
                text="follow-up",
                platform="telegram",
                user_id="tg-42",
            )
            assert "follow-up" in result2.response
            assert router.get_session_id("chat-1", "alice") == result2.session_id
