"""CLI entry point for the self-reflection engine.

Subcommands (flag-based, matching `hermes reflect <agent> [flags]`):

- plain             run a DRY-RUN reflection for <agent>
- --apply           run reflection and apply approved changes
- --list            print the last 20 reflection_log entries for <agent>
- --rollback TS     roll back the change at timestamp TS for <agent>
- --status          show reflection status across all agents
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fmt_ts(ts: float) -> str:
    try:
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(float(ts)))
    except (TypeError, ValueError):
        return str(ts)


def _log_path(agent: str) -> Path:
    from hermes_cli.agent_registry import get_agent_dir
    return get_agent_dir(agent) / "reflection_log.jsonl"


def _read_log(agent: str) -> List[Dict[str, Any]]:
    path = _log_path(agent)
    if not path.is_file():
        return []
    out: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except Exception:
            continue
    return out


def _outcome_counts(agent: str, lookback_days: int = 7) -> Dict[str, int]:
    from hermes_constants import get_hermes_home
    db_path = get_hermes_home() / "state.db"
    if not db_path.exists():
        return {"positive": 0, "negative": 0, "neutral": 0}
    cutoff = time.time() - lookback_days * 86400
    con = sqlite3.connect(str(db_path))
    try:
        rows = con.execute(
            """SELECT outcome, COUNT(*) FROM sessions
               WHERE started_at >= ? AND (source LIKE ? OR 1=1)
               GROUP BY outcome""",
            (cutoff, f"%{agent}%"),
        ).fetchall()
    finally:
        con.close()
    counts = {"positive": 0, "negative": 0, "neutral": 0}
    for outcome, n in rows:
        if outcome in ("positive", "negative"):
            counts[outcome] = int(n)
        else:
            counts["neutral"] += int(n)
    return counts


# ---------------------------------------------------------------------------
# Subcommand bodies
# ---------------------------------------------------------------------------


def _run(agent: str, lookback: int, apply_changes: bool) -> int:
    from cron.reflection import run_reflection
    result = asyncio.run(
        run_reflection(agent, lookback_days=lookback, dry_run=not apply_changes)
    )
    print(f"Agent: {agent}")
    print(f"Confidence: {result.overall_confidence:.2f}")
    print(f"Memory proposals ({len(result.memory_proposals)}):")
    for mp in result.memory_proposals:
        print(f"  [{mp.get('action')}] {mp.get('content', '')[:120]}")
        reason = (mp.get("reason") or "")[:120]
        if reason:
            print(f"    reason: {reason}")
    print(f"Skill proposals ({len(result.skill_proposals)}):")
    for sp in result.skill_proposals:
        print(f"  [{sp.get('action')}] {sp.get('name')} ({sp.get('priority')})")
        reason = (sp.get("reason") or "")[:120]
        if reason:
            print(f"    reason: {reason}")
    print(f"Cross-agent observations ({len(result.cross_agent_observations)}):")
    for obs in result.cross_agent_observations:
        agents = ", ".join(obs.get("relevant_agents") or [])
        print(f"  → [{agents}] {obs.get('observation', '')[:120]}")
    print(f"Skill-gap queries: {result.skill_gap_queries}")
    mode = "APPLIED" if apply_changes else "DRY-RUN"
    print(f"Mode: {mode}")
    return 0


def _list(agent: str) -> int:
    entries = _read_log(agent)[-20:]
    if not entries:
        print(f"No reflection log entries for {agent}.")
        return 0
    print(f"Last {len(entries)} reflection_log entries for {agent}:")
    for e in entries:
        ts = _fmt_ts(e.get("timestamp", 0))
        action = e.get("action", "?")
        applied = "applied" if e.get("applied") else "skipped"
        data = e.get("data") or {}
        brief = ""
        if isinstance(data, dict):
            brief = (
                data.get("content")
                or data.get("name")
                or data.get("observation")
                or data.get("query")
                or e.get("reason", "")
            )
            if isinstance(brief, str):
                brief = brief[:100]
        print(f"  {ts}  {action:24}  {applied:8}  {brief}")
    return 0


def _rollback(agent: str, target_ts: str) -> int:
    entries = _read_log(agent)
    match: Optional[Dict[str, Any]] = None
    for e in entries:
        if str(e.get("timestamp", "")).startswith(str(target_ts)) or \
           str(int(e.get("timestamp", 0))) == str(int(float(target_ts))):
            match = e
            break
    if match is None:
        print(f"No log entry found for timestamp {target_ts}.")
        return 1
    action = match.get("action")
    data = match.get("data") or {}
    if action == "memory_add":
        from hermes_cli.agent_registry import get_agent_dir
        memory_path = get_agent_dir(agent) / "MEMORY.md"
        target = data.get("content", "")
        if memory_path.is_file() and target:
            existing = memory_path.read_text(encoding="utf-8")
            line_prefix = f"- {target}"
            lines = existing.splitlines()
            new_lines = [l for l in lines if target not in l]
            if len(new_lines) != len(lines):
                memory_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
                print(f"Rolled back memory_add from {_fmt_ts(match['timestamp'])}")
            else:
                print("Entry already removed from MEMORY.md; logging rollback only.")
        else:
            print("MEMORY.md missing; logging rollback only.")
    elif action == "cross_agent_capture":
        print("Cross-agent captures cannot be un-captured from HIPP0 "
              "(staleness will handle it); logging retraction only.")
    else:
        print(f"Action '{action}' has no physical rollback; logging only.")

    rollback_entry = {
        "timestamp": time.time(),
        "action": "rollback",
        "data": {"target_timestamp": match.get("timestamp"),
                 "target_action": action},
        "applied": True,
    }
    with open(_log_path(agent), "a", encoding="utf-8") as f:
        f.write(json.dumps(rollback_entry) + "\n")
    return 0


def _status() -> int:
    from hermes_cli.agent_registry import list_agents
    agents = list_agents()
    print(f"Reflection status across {len(agents)} agents:\n")
    print(f"{'Agent':12}  {'Last run':19}  {'Applied':>7}  {'Pending':>7}  "
          f"{'+':>3} {'-':>3} {'o':>3}")
    print("-" * 72)
    for agent in agents:
        entries = _read_log(agent)
        last_run = "(never)"
        applied_count = 0
        pending = 0
        if entries:
            # find last reflection_complete
            for e in reversed(entries):
                if e.get("action") == "reflection_complete":
                    last_run = _fmt_ts(e.get("timestamp", 0))
                    break
            # count applied memory in most recent cycle (since last reflection_complete)
            cycle: List[Dict[str, Any]] = []
            for e in reversed(entries):
                if e.get("action") == "reflection_complete":
                    break
                cycle.append(e)
            applied_count = sum(
                1 for e in cycle
                if e.get("applied") and e.get("action", "").startswith("memory_")
            )
            pending = sum(
                1 for e in entries
                if e.get("action") == "skill_proposal" and not e.get("applied")
            )
        counts = _outcome_counts(agent, lookback_days=7)
        print(f"{agent:12}  {last_run:19}  {applied_count:>7}  {pending:>7}  "
              f"{counts['positive']:>3} {counts['negative']:>3} {counts['neutral']:>3}")
    return 0


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


def reflect_command(args) -> int:
    if args.status:
        return _status()
    if not args.agent:
        print("error: agent name required (use --status for overview)",
              file=sys.stderr)
        return 2
    if args.list:
        return _list(args.agent)
    if args.rollback:
        return _rollback(args.agent, args.rollback)
    return _run(args.agent, args.lookback, apply_changes=args.apply)
