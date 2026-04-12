#!/usr/bin/env python3
"""talk-to-alice CLI REPL — persistent agent conversation via HIPP0.

Usage:
    python hermes_cli/repl.py --agent alice

Reads ANTHROPIC_API_KEY from environment and HIPP0 API key from
/etc/team-hippo/api-key.txt. Starts a HIPP0 session, compiles
context, then runs a stdin/stdout conversation loop against the
named agent's SOUL.md persona.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Talk to a persistent Hermes agent via HIPP0"
    )
    parser.add_argument(
        "--agent", default="alice", help="Agent name (default: alice)"
    )
    args = parser.parse_args()

    # ---- preflight checks ----

    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not anthropic_key:
        print("error: ANTHROPIC_API_KEY environment variable is not set.", file=sys.stderr)
        sys.exit(1)

    hipp0_key_path = "/etc/team-hippo/api-key.txt"
    try:
        hipp0_key = open(hipp0_key_path).read().strip()
    except FileNotFoundError:
        print(f"error: {hipp0_key_path} not found.", file=sys.stderr)
        sys.exit(1)

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

    # ---- HIPP0 session setup ----

    provider = Hipp0MemoryProvider(
        base_url="http://127.0.0.1:3100",
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
        # Compile failure is non-fatal — degraded mode
        from agent.hipp0_memory_provider import CompiledContext
        compiled = CompiledContext(degraded=True, degraded_reason="compile failed at REPL startup")

    system_prompt = build_slim_system_prompt(
        profile.soul,
        compiled_context_block=compiled.as_prompt_block(),
        platform_hint="cli",
    )

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
        skip_memory=True,
        slim_prompt=True,
    )

    # ---- conversation loop ----

    transcript_lines: list[str] = []

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
            transcript_lines.append(f"USER: {user_input}")
            transcript_lines.append(f"ASSISTANT({args.agent}): {response}")
    except KeyboardInterrupt:
        print()

    # ---- cleanup: capture transcript + end session ----

    if transcript_lines:
        try:
            loop.run_until_complete(
                provider.capture("\n".join(transcript_lines), source="hermes")
            )
        except Exception:
            pass  # WAL handles failures

    try:
        loop.run_until_complete(provider.end_session())
    except Exception:
        pass

    loop.run_until_complete(provider.aclose())
    loop.close()

    print("Goodbye.")


if __name__ == "__main__":
    main()
