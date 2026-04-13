#!/usr/bin/env python3
"""talk-to-alice CLI REPL — persistent agent conversation via HIPP0.

Usage:
    python hermes_cli/repl.py --agent alice

Reads ANTHROPIC_API_KEY from environment and HIPP0 API key from
HIPP0_API_KEY_FILE (default /etc/team-hippo/api-key.txt).  Starts a
HIPP0 session, compiles context, then runs a stdin/stdout conversation
loop against the named agent's SOUL.md persona.

Memory integration:
  - Builtin memory/user tools are enabled (MEMORY.md / USER.md on disk).
  - Hipp0MemoryProvider is registered as an external memory provider so
    AIAgent's conversation loop calls compile (prefetch) before each turn
    and capture (sync) after each turn automatically.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sync adapter — bridges async Hipp0MemoryProvider into the sync
# MemoryProvider ABC that AIAgent's MemoryManager expects.
# ---------------------------------------------------------------------------

class _Hipp0SyncAdapter:
    """Thin sync wrapper around Hipp0MemoryProvider for MemoryManager.

    MemoryManager calls prefetch() and sync_turn() synchronously from
    within AIAgent's blocking conversation loop.  We bridge to the
    async provider via the REPL's event loop.
    """

    def __init__(self, provider: Any, loop: asyncio.AbstractEventLoop) -> None:
        self._provider = provider
        self._loop = loop

    # -- MemoryProvider ABC required -----------------------------------------

    @property
    def name(self) -> str:
        return "hipp0"

    def is_available(self) -> bool:
        return self._provider.is_available()

    def initialize(self, session_id: str, **kwargs: Any) -> None:
        self._provider.initialize(session_id, **kwargs)

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return []

    # -- Per-turn hooks (the whole point of this adapter) --------------------

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        """Compile HIPP0 context for the upcoming turn."""
        if not query:
            return ""
        try:
            compiled = self._loop.run_until_complete(
                self._provider.compile(query, fast_mode=True)
            )
            return compiled.as_prompt_block()
        except Exception as e:
            logger.warning("HIPP0 prefetch (compile) failed: %s", e)
            return ""

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        pass  # No background prefetch — we compile synchronously in prefetch()

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        """Capture the completed turn to HIPP0."""
        transcript = f"USER: {user_content}\nASSISTANT: {assistant_content}"
        try:
            self._loop.run_until_complete(
                self._provider.capture(transcript, source="hermes")
            )
        except Exception as e:
            logger.warning("HIPP0 sync_turn (capture) failed: %s", e)

    # -- Optional hooks (no-op defaults) -------------------------------------

    def system_prompt_block(self) -> str:
        return ""

    def shutdown(self) -> None:
        pass

    def on_turn_start(self, turn_number: int, message: str, **kwargs: Any) -> None:
        pass

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        pass

    def on_pre_compress(self, messages: List[Dict[str, Any]]) -> str:
        return ""

    def on_memory_write(self, action: str, target: str, content: str) -> None:
        pass

    def on_delegation(self, task: str, result: str, *, child_session_id: str = "", **kwargs: Any) -> None:
        pass

    def get_config_schema(self) -> List[Dict[str, Any]]:
        return []

    def save_config(self, values: Dict[str, Any], hermes_home: str) -> None:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Talk to a persistent Hermes agent via HIPP0"
    )
    parser.add_argument(
        "--agent", default="hipp0", help="Agent name (default: hipp0)"
    )
    args = parser.parse_args()

    # ---- preflight checks ----

    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not anthropic_key:
        print("error: ANTHROPIC_API_KEY environment variable is not set.", file=sys.stderr)
        sys.exit(1)

    hipp0_key_path = os.environ.get(
        "HIPP0_API_KEY_FILE", "/etc/team-hippo/api-key.txt"
    )
    try:
        hipp0_key = open(hipp0_key_path).read().strip()
    except FileNotFoundError:
        print(f"error: {hipp0_key_path} not found.", file=sys.stderr)
        sys.exit(1)

    hipp0_base_url = os.environ.get("HIPP0_BASE_URL", "http://127.0.0.1:3100")

    # Ensure Hermes code can be imported from repo root
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    from hermes_cli.agent_registry import AgentNotFoundError, get_agent
    from agent.hipp0_memory_provider import Hipp0MemoryProvider
    from agent.prompt_builder import build_slim_system_prompt

    try:
        profile = get_agent(args.agent)
    except AgentNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)

    if not profile.config.project_id:
        print(
            f"error: agent {args.agent!r} has no project_id in config.yaml.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Ensure builtin memory directories exist
    hermes_home = Path.home() / ".hermes"
    (hermes_home / "memories").mkdir(parents=True, exist_ok=True)

    # ---- HIPP0 session setup ----

    provider = Hipp0MemoryProvider(
        base_url=hipp0_base_url,
        api_key=hipp0_key,
        project_id=str(profile.config.project_id),
        agent_name=profile.name,
        agent_id=str(profile.config.agent_id or ""),
        pending_wal_path=profile.pending_wal_path,
        memory_md_path=profile.memory_path,
    )

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        session_id = loop.run_until_complete(
            provider.start_session(platform="cli")
        )
    except Exception as e:
        print(f"error: failed to start HIPP0 session: {e}", file=sys.stderr)
        loop.run_until_complete(provider.aclose())
        loop.close()
        sys.exit(1)

    try:
        compiled = loop.run_until_complete(
            provider.compile("General conversation", fast_mode=False)
        )
    except Exception:
        from agent.hipp0_memory_provider import CompiledContext
        compiled = CompiledContext(degraded=True, degraded_reason="compile failed at REPL startup")

    # The era-1 compile endpoint has a hardcoded MIN_SCORE=0.5 that filters
    # out all decisions (cosine scores top out at ~0.11).  As a fallback,
    # fetch recent decisions directly and inject them as supplementary
    # context so the agent has real project knowledge on day one.
    decisions_block = ""
    if not compiled.decisions:
        try:
            import httpx as _httpx
            _resp = _httpx.get(
                f"{hipp0_base_url}/api/decisions",
                params={
                    "project_id": str(profile.config.project_id),
                    "limit": "20",
                },
                headers={"Authorization": f"Bearer {hipp0_key}"},
                timeout=10,
            )
            if _resp.status_code == 200:
                _decisions = _resp.json()
                if _decisions:
                    _lines = ["## Project decisions (from HIPP0 memory)", ""]
                    for _d in _decisions:
                        _title = _d.get("title", "untitled")
                        _desc = (_d.get("description") or "")[:300]
                        _by = _d.get("made_by", "?")
                        _lines.append(f"- **{_title}** (by {_by}): {_desc}")
                    decisions_block = "\n".join(_lines)
        except Exception as e:
            logger.debug("Direct decisions fetch failed: %s", e)

    # Cross-session memory fallback: fetch recent captures (raw conversation
    # transcripts) for this agent so Alice can remember what was said in
    # previous sessions even before the distillery extracts structured facts.
    captures_block = ""
    try:
        import httpx as _httpx_cap
        _cap_resp = _httpx_cap.get(
            f"{hipp0_base_url}/api/hermes/captures",
            params={
                "project_id": str(profile.config.project_id),
                "limit": "20",
            },
            headers={"Authorization": f"Bearer {hipp0_key}"},
            timeout=10,
        )
        if _cap_resp.status_code == 200:
            _captures = _cap_resp.json()
            if _captures:
                _cap_lines = ["## Recent conversations (from HIPP0 memory)", ""]
                for _cap in _captures:
                    _text = (_cap.get("conversation_text") or "")[:500]
                    _ts = _cap.get("created_at", "unknown")
                    if _text.strip():
                        _cap_lines.append(f"### Session ({_ts})")
                        _cap_lines.append(_text)
                        _cap_lines.append("")
                captures_block = "\n".join(_cap_lines)
    except Exception as e:
        logger.debug("Captures fallback fetch failed: %s", e)

    # Sync user_facts from HIPP0 to local USER.md so the agent profile
    # stays current even when the compile API is slow or down.
    user_facts_block = ""
    _uf_facts: list = []
    try:
        import httpx as _httpx_uf
        # Query the lightweight extracted-facts endpoint (distillery-extracted
        # user_facts from the user_facts table).
        _uf_resp = _httpx_uf.get(
            f"{hipp0_base_url}/api/hermes/extracted-facts",
            params={
                "project_id": str(profile.config.project_id),
                "external_user_id": "owner",
            },
            headers={"Authorization": f"Bearer {hipp0_key}"},
            timeout=10,
        )
        if _uf_resp.status_code == 200:
            _uf_data = _uf_resp.json()
            _uf_facts = _uf_data.get("facts", [])
    except Exception as e:
        logger.debug("Extracted user-facts fetch failed: %s", e)

    if _uf_facts:
        # Build user_facts section for system prompt
        _uf_lines = ["## User Preferences (from HIPP0 memory)", ""]
        for _f in _uf_facts:
            _key = _f.get("key", "unknown")
            _val = _f.get("value", "")
            if _val:
                _uf_lines.append(f"- **{_key}**: {_val}")
        user_facts_block = "\n".join(_uf_lines)

        # Write to USER.md for local persistence
        user_md_path = profile.root / "USER.md"
        try:
            existing_content = user_md_path.read_text() if user_md_path.exists() else ""
            # Replace or append the HIPP0 section
            marker_start = "### HIPP0 Extracted User Facts"
            marker_end = "### END HIPP0 Facts"
            facts_section = f"{marker_start}\n"
            for _f in _uf_facts:
                _key = _f.get("key", "unknown")
                _val = _f.get("value", "")
                if _val:
                    facts_section += f"- **{_key}**: {_val}\n"
            facts_section += f"{marker_end}\n"

            if marker_start in existing_content:
                import re
                existing_content = re.sub(
                    rf"{re.escape(marker_start)}.*?{re.escape(marker_end)}\n?",
                    facts_section,
                    existing_content,
                    flags=re.DOTALL,
                )
            else:
                existing_content = existing_content.rstrip() + "\n\n" + facts_section

            user_md_path.write_text(existing_content)
        except Exception as e:
            logger.debug("Failed to write USER.md: %s", e)

    extra_sections = []
    if decisions_block:
        extra_sections.append(decisions_block)
    if captures_block:
        extra_sections.append(captures_block)
    if user_facts_block:
        extra_sections.append(user_facts_block)

    system_prompt = build_slim_system_prompt(
        profile.soul,
        compiled_context_block=compiled.as_prompt_block(),
        platform_hint="cli",
        extra_sections=extra_sections if extra_sections else None,
    )

    # ---- Session DB for session_search tool ----

    from hermes_state import SessionDB

    session_db = SessionDB()  # defaults to ~/.hermes/state.db

    # ---- AIAgent setup ----

    from run_agent import AIAgent

    agent = AIAgent(
        model=profile.config.model,
        provider="anthropic",
        api_key=anthropic_key,
        max_iterations=int(profile.config.extra.get("max_iterations", 50)),
        quiet_mode=True,
        ephemeral_system_prompt=system_prompt,
        platform="cli",
        skip_context_files=True,
        skip_memory=False,  # Enable builtin memory/user tools + memory manager
        slim_prompt=True,
        session_db=session_db,
        agent_name=profile.name,
    )

    # ---- Wire HIPP0 as external memory provider ----
    # AIAgent's __init__ creates _memory_manager only if a named plugin is
    # configured.  We bypass that and inject the HIPP0 adapter directly so
    # prefetch (compile) runs before each turn and sync_turn (capture) runs
    # after each turn.

    from agent.memory_manager import MemoryManager

    adapter = _Hipp0SyncAdapter(provider, loop)
    if agent._memory_manager is None:
        agent._memory_manager = MemoryManager()
    agent._memory_manager.add_provider(adapter)
    adapter.initialize(session_id, platform="cli", hermes_home=str(hermes_home))

    # ---- conversation loop ----

    try:
        while True:
            try:
                user_input = input("you> ")
            except EOFError:
                print()
                break
            except KeyboardInterrupt:
                print()
                break

            if not user_input.strip():
                continue

            try:
                response = agent.chat(user_input)
            except KeyboardInterrupt:
                print("\n(interrupted)")
                break
            except Exception as e:
                print(f"error: {e}", file=sys.stderr)
                continue

            print(f"{args.agent}> {response}")
    except KeyboardInterrupt:
        print()

    # ---- cleanup: end HIPP0 session ----

    try:
        loop.run_until_complete(provider.end_session())
    except Exception:
        pass

    loop.run_until_complete(provider.aclose())
    loop.close()

    print("Goodbye.")


if __name__ == "__main__":
    main()
