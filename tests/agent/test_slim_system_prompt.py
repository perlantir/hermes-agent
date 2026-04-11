"""Tests for the slim-prompt path in :mod:`agent.prompt_builder`.

Exercises :func:`build_slim_system_prompt` and asserts the H4 size
contract from the ``feat/persistent-agents-hipp0`` brief:

  - slim prompt is well under 1500 tokens (~6000 chars) for a
    realistic SOUL.md + compiled context.
  - a representative "full" prompt assembled from the existing
    guidance constants is clearly above 3000 tokens.
"""

from __future__ import annotations

import pytest

from agent.prompt_builder import (
    DEFAULT_AGENT_IDENTITY,
    GOOGLE_MODEL_OPERATIONAL_GUIDANCE,
    MEMORY_GUIDANCE,
    OPENAI_MODEL_EXECUTION_GUIDANCE,
    PLATFORM_HINTS,
    SESSION_SEARCH_GUIDANCE,
    SKILLS_GUIDANCE,
    SLIM_PROMPT_TARGET_TOKENS,
    TOOL_USE_ENFORCEMENT_GUIDANCE,
    build_slim_system_prompt,
    estimate_prompt_tokens,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


REALISTIC_SOUL = """# Alice — Sales Lead

You are **Alice**, the sales lead for the Hermes/HIPP0 platform.

## Role

- Qualify inbound prospects and understand their agent/memory needs.
- Track commitments, next steps, and decisions across every conversation.
- Remember user preferences and apply them without re-asking.

## Voice

- Warm, direct, and concise. Never salesy.
- Lead with the user's stated goal, not product features.

## Operating rules

- Treat anything the user tells you as a durable fact.
- State the exact next step in writing at the end of each turn.
- If the user corrects you, accept the correction as ground truth.
"""

REALISTIC_COMPILED_BLOCK = """## Compiled context

- [dec-1] User prefers phone calls to emails (score=0.91)
- [dec-2] Previously mentioned interest in the HIPP0 memory demo (score=0.78)
- [dec-3] Based in Pacific timezone (score=0.70)
"""


# ---------------------------------------------------------------------------
# build_slim_system_prompt
# ---------------------------------------------------------------------------


class TestBuildSlimSystemPrompt:
    def test_soul_only(self):
        out = build_slim_system_prompt(REALISTIC_SOUL)
        assert "# Alice" in out
        assert "Sales Lead" in out

    def test_includes_compiled_context(self):
        out = build_slim_system_prompt(
            REALISTIC_SOUL,
            compiled_context_block=REALISTIC_COMPILED_BLOCK,
        )
        assert "## Compiled context" in out
        assert "dec-1" in out
        assert "prefers phone calls" in out

    def test_platform_hint_telegram(self):
        out = build_slim_system_prompt(
            REALISTIC_SOUL,
            platform_hint="telegram",
        )
        assert "Telegram" in out

    def test_platform_hint_unknown_is_ignored(self):
        out = build_slim_system_prompt(
            REALISTIC_SOUL,
            platform_hint="mars-chat",
        )
        # Soul still there; no exception, no garbage suffix.
        assert "# Alice" in out
        assert "mars-chat" not in out

    def test_time_hint_appended(self):
        out = build_slim_system_prompt(
            REALISTIC_SOUL,
            time_hint="Conversation started: Monday, April 11, 2026 10:00 AM",
        )
        assert "April 11, 2026" in out

    def test_extra_sections(self):
        out = build_slim_system_prompt(
            REALISTIC_SOUL,
            extra_sections=["Extra note: be extra concise"],
        )
        assert "Extra note" in out

    def test_rejects_non_string_soul(self):
        with pytest.raises(TypeError):
            build_slim_system_prompt(None)  # type: ignore[arg-type]

    def test_empty_extras_are_filtered(self):
        out = build_slim_system_prompt(
            REALISTIC_SOUL,
            extra_sections=["", None, "real"],  # type: ignore[list-item]
        )
        assert "real" in out
        # Make sure we didn't produce stray "None" literals.
        assert "None" not in out.split("\n")


# ---------------------------------------------------------------------------
# Size contract: slim < 1500, full > 3000
# ---------------------------------------------------------------------------


class TestSizeContract:
    def test_slim_prompt_under_target(self):
        slim = build_slim_system_prompt(
            REALISTIC_SOUL,
            compiled_context_block=REALISTIC_COMPILED_BLOCK,
            platform_hint="telegram",
            time_hint="Conversation started: Monday, April 11, 2026 10:00 AM",
        )
        tokens = estimate_prompt_tokens(slim)
        assert tokens < SLIM_PROMPT_TARGET_TOKENS, (
            f"slim prompt is {tokens} tokens, expected < {SLIM_PROMPT_TARGET_TOKENS}"
        )
        # Should still be non-trivial — don't let a regression produce
        # an empty prompt and pass this test.
        assert tokens > 50

    def test_full_prompt_significantly_larger(self):
        # Representative "full" prompt: every guidance block + a SOUL +
        # a realistic context-file section (using AGENTS.md as a stand-
        # in, which is what the non-slim path loads via
        # build_context_files_prompt). This models what
        # AIAgent._build_system_prompt produces when all tool guidance
        # is active and the agent is running in a project with a
        # normal-size AGENTS.md.
        from pathlib import Path

        repo_root = Path(__file__).resolve().parents[2]
        agents_md = (repo_root / "AGENTS.md").read_text(encoding="utf-8")

        full_parts = [
            DEFAULT_AGENT_IDENTITY,
            REALISTIC_SOUL,
            MEMORY_GUIDANCE,
            SESSION_SEARCH_GUIDANCE,
            SKILLS_GUIDANCE,
            TOOL_USE_ENFORCEMENT_GUIDANCE,
            OPENAI_MODEL_EXECUTION_GUIDANCE,
            GOOGLE_MODEL_OPERATIONAL_GUIDANCE,
            PLATFORM_HINTS["telegram"],
            REALISTIC_COMPILED_BLOCK,
            agents_md,
            "Conversation started: Monday, April 11, 2026 10:00 AM",
        ]
        full = "\n\n".join(full_parts)
        full_tokens = estimate_prompt_tokens(full)
        assert full_tokens > 3000, (
            f"representative full prompt is {full_tokens} tokens, "
            f"expected > 3000"
        )

        # The slim path MUST be materially smaller than the full path.
        slim = build_slim_system_prompt(
            REALISTIC_SOUL,
            compiled_context_block=REALISTIC_COMPILED_BLOCK,
            platform_hint="telegram",
        )
        slim_tokens = estimate_prompt_tokens(slim)
        assert slim_tokens * 2 < full_tokens, (
            f"slim ({slim_tokens}) should be less than half of full "
            f"({full_tokens}) tokens"
        )


# ---------------------------------------------------------------------------
# estimate_prompt_tokens sanity
# ---------------------------------------------------------------------------


class TestEstimateTokens:
    def test_zero_for_empty(self):
        assert estimate_prompt_tokens("") == 0

    def test_roughly_chars_over_four(self):
        text = "x" * 4000
        assert estimate_prompt_tokens(text) == 1000

    def test_non_empty_min_one(self):
        assert estimate_prompt_tokens("xyz") == 1
