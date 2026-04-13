"""Daily self-reflection engine for persistent agents.

Runs on a stagger-scheduled daily cron per agent. Gathers the last N
days of session outcomes + tool usage + memory, sends a structured
analysis prompt to Claude Haiku via the existing auxiliary_client
infrastructure, and applies confidence-gated improvements:

- Memory proposals tagged "[auto-reflection]" are appended/replaced in
  MEMORY.md (max :data:`MAX_MEMORY_PER_CYCLE` per run).
- Skill proposals are NEVER auto-applied — logged for user review.
- Cross-agent observations are captured to HIPP0 via the normal
  ``/api/capture`` pipeline with ``source="reflection"``.

All actions append-only logged to ``~/.hermes/agents/<name>/reflection_log.jsonl``.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Confidence thresholds and caps
# ---------------------------------------------------------------------------

CONFIDENCE_THRESHOLDS: Dict[str, float] = {
    "memory_add": 0.5,
    "memory_replace": 0.7,
    "skill_create": 1.0,
    "skill_update": 1.0,
    "skill_search_hub": 0.3,
    "cross_agent": 0.6,
}

MAX_MEMORY_PER_CYCLE = 3
MAX_SKILLS_PER_CYCLE = 0

AUTO_REFLECTION_TAG = "[auto-reflection]"
MIN_OVERALL_CONFIDENCE = 0.5
MIN_SESSIONS_REQUIRED = 3
DEFAULT_LOOKBACK_DAYS = 7
HAIKU_MODEL = "claude-haiku-4-5-20251001"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ReflectionInput:
    agent_name: str
    recent_sessions: List[Dict[str, Any]] = field(default_factory=list)
    positive_sessions: List[Dict[str, Any]] = field(default_factory=list)
    negative_sessions: List[Dict[str, Any]] = field(default_factory=list)
    current_skills: List[str] = field(default_factory=list)
    memory_snapshot: str = ""
    user_snapshot: str = ""
    tool_usage: Dict[str, int] = field(default_factory=dict)
    compiled_context: Optional[str] = None
    lookback_days: int = DEFAULT_LOOKBACK_DAYS


@dataclass
class ReflectionOutput:
    skill_proposals: List[Dict[str, Any]] = field(default_factory=list)
    memory_proposals: List[Dict[str, Any]] = field(default_factory=list)
    cross_agent_observations: List[Dict[str, Any]] = field(default_factory=list)
    skill_gap_queries: List[str] = field(default_factory=list)
    overall_confidence: float = 0.0


# ---------------------------------------------------------------------------
# Filesystem helpers
# ---------------------------------------------------------------------------


def _agent_dir(agent_name: str) -> Path:
    from hermes_cli.agent_registry import get_agent_dir
    return get_agent_dir(agent_name)


def _reflection_log_path(agent_name: str) -> Path:
    return _agent_dir(agent_name) / "reflection_log.jsonl"


def _append_log(agent_name: str, entry: Dict[str, Any]) -> None:
    path = _reflection_log_path(agent_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = dict(entry)
    entry.setdefault("timestamp", time.time())
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Data gathering — no LLM call
# ---------------------------------------------------------------------------


def _state_db_path() -> Path:
    from hermes_constants import get_hermes_home
    return get_hermes_home() / "state.db"


def _query_sessions(
    agent_name: str,
    lookback_days: int,
) -> Dict[str, List[Dict[str, Any]]]:
    """Return sessions for *agent_name* in the lookback window, split by outcome.

    Sessions are filtered by the ``agent_name`` column (set when an
    AIAgent is constructed for a persistent agent). Rows with a NULL
    ``agent_name`` are legacy/cross-flow sessions written before this
    column existed — we include them as a best-effort fallback so old
    sessions still show up in every agent's reflection rather than
    vanishing entirely.
    """
    cutoff = time.time() - lookback_days * 86400
    db_path = _state_db_path()
    if not db_path.exists():
        return {"all": [], "positive": [], "negative": [], "neutral": []}
    con = sqlite3.connect(str(db_path))
    con.row_factory = sqlite3.Row
    try:
        rows = con.execute(
            """SELECT id, source, agent_name, started_at, ended_at, message_count,
                      outcome, outcome_source, outcome_detail
               FROM sessions
               WHERE started_at >= ?
                 AND (agent_name = ? OR agent_name IS NULL)
               ORDER BY started_at DESC
               LIMIT 500""",
            (cutoff, agent_name),
        ).fetchall()
        pool = [dict(r) for r in rows]
        # Attach the first user message as a summary
        for s in pool:
            msg_row = con.execute(
                """SELECT content FROM messages
                   WHERE session_id = ? AND role = 'user'
                   ORDER BY timestamp ASC LIMIT 1""",
                (s["id"],),
            ).fetchone()
            s["first_user_message"] = (msg_row["content"] or "")[:200] if msg_row else ""
        return {
            "all": pool,
            "positive": [s for s in pool if s.get("outcome") == "positive"],
            "negative": [s for s in pool if s.get("outcome") == "negative"],
            "neutral": [s for s in pool if not s.get("outcome")],
        }
    finally:
        con.close()


def _query_tool_usage(agent_name: str, lookback_days: int) -> Dict[str, int]:
    cutoff = time.time() - lookback_days * 86400
    db_path = _state_db_path()
    if not db_path.exists():
        return {}
    con = sqlite3.connect(str(db_path))
    try:
        rows = con.execute(
            """SELECT m.tool_name, COUNT(*) as n
               FROM messages m JOIN sessions s ON m.session_id = s.id
               WHERE s.started_at >= ?
                 AND (s.agent_name = ? OR s.agent_name IS NULL)
                 AND m.tool_name IS NOT NULL
               GROUP BY m.tool_name
               ORDER BY n DESC
               LIMIT 50""",
            (cutoff, agent_name),
        ).fetchall()
        return {r[0]: int(r[1]) for r in rows}
    finally:
        con.close()


def _read_text_file(path: Path, limit: int = 4000) -> str:
    try:
        data = path.read_text(encoding="utf-8")
        return data[:limit]
    except FileNotFoundError:
        return ""
    except Exception as exc:
        logger.debug("read %s failed: %s", path, exc)
        return ""


def _list_skills(agent_name: str) -> List[str]:
    skills_dir = _agent_dir(agent_name) / "skills"
    if not skills_dir.is_dir():
        return []
    return sorted(
        p.name for p in skills_dir.iterdir()
        if p.is_dir() and (p / "SKILL.md").is_file()
    )


async def _try_compile_context(agent_name: str) -> Optional[str]:
    """Best-effort call to HIPP0 compile for self-improvement context."""
    try:
        from hermes_cli.agent_registry import get_agent
        profile = get_agent(agent_name)
        project_id = getattr(profile.config, "project_id", None)
        if not project_id:
            return None
        base_url = os.environ.get("HIPP0_BASE_URL", "http://127.0.0.1:3100")
        api_key = os.environ.get("HIPP0_API_KEY", "")
        if not api_key:
            key_file = os.environ.get("HIPP0_API_KEY_FILE")
            if key_file and Path(key_file).is_file():
                api_key = Path(key_file).read_text(encoding="utf-8").strip()
        if not api_key:
            return None
        import httpx
        async with httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=10.0,
        ) as client:
            resp = await client.post(
                "/api/hermes/compile",
                json={
                    "agent_name": agent_name,
                    "project_id": str(project_id),
                    "task": "self-improvement analysis",
                    "max_tokens": 2000,
                },
            )
            if resp.status_code != 200:
                return None
            data = resp.json()
            return data.get("compiled_context") or data.get("context") or None
    except Exception as exc:
        logger.debug("compile call failed: %s", exc)
        return None


def gather_reflection_input(
    agent_name: str,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    *,
    include_compile: bool = True,
) -> ReflectionInput:
    sessions = _query_sessions(agent_name, lookback_days)
    tool_usage = _query_tool_usage(agent_name, lookback_days)
    skills = _list_skills(agent_name)
    memory = _read_text_file(_agent_dir(agent_name) / "MEMORY.md", limit=4000)
    user_md = _read_text_file(_agent_dir(agent_name) / "USER.md", limit=2000)
    compiled: Optional[str] = None
    if include_compile:
        try:
            compiled = asyncio.get_event_loop().run_until_complete(
                _try_compile_context(agent_name)
            )
        except RuntimeError:
            compiled = None
    return ReflectionInput(
        agent_name=agent_name,
        recent_sessions=sessions["all"],
        positive_sessions=sessions["positive"],
        negative_sessions=sessions["negative"],
        current_skills=skills,
        memory_snapshot=memory,
        user_snapshot=user_md,
        tool_usage=tool_usage,
        compiled_context=compiled,
        lookback_days=lookback_days,
    )


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def _fmt_session_line(s: Dict[str, Any]) -> str:
    ts = time.strftime("%Y-%m-%d %H:%M", time.localtime(s.get("started_at") or 0))
    first = (s.get("first_user_message") or "").replace("\n", " ")
    detail = s.get("outcome_detail") or ""
    return f"- {ts} | {first[:180]} | detail={detail[:120]}"


def _build_reflection_prompt(rin: ReflectionInput) -> str:
    total = len(rin.recent_sessions)
    pos = len(rin.positive_sessions)
    neg = len(rin.negative_sessions)
    neutral = max(0, total - pos - neg)
    neg_lines = "\n".join(_fmt_session_line(s) for s in rin.negative_sessions[:10]) or "(none)"
    pos_lines = "\n".join(_fmt_session_line(s) for s in rin.positive_sessions[:10]) or "(none)"
    tool_lines = "\n".join(
        f"  {name}: {count}"
        for name, count in sorted(rin.tool_usage.items(), key=lambda x: -x[1])[:20]
    ) or "  (no tool usage recorded)"
    skills_list = ", ".join(rin.current_skills[:50]) or "(none)"
    return f"""You are analyzing the recent performance of agent "{rin.agent_name}" to identify concrete improvements.

RECENT SESSIONS (last {rin.lookback_days} days):
- Total: {total}
- Positive outcomes: {pos}
- Negative outcomes: {neg}
- No outcome recorded: {neutral}

NEGATIVE OUTCOME SESSIONS (what went wrong — analyze these carefully):
{neg_lines}

POSITIVE OUTCOME SESSIONS (what went right — learn from these):
{pos_lines}

TOOL USAGE (last {rin.lookback_days} days):
{tool_lines}

CURRENT SKILLS ({len(rin.current_skills)}):
{skills_list}

CURRENT MEMORY ({len(rin.memory_snapshot)} chars):
{rin.memory_snapshot[:2000]}

Respond with ONLY this JSON (no markdown fences, no preamble):
{{
  "skill_proposals": [
    {{"action": "create|update|search_hub", "name": "skill-name", "reason": "why this would help based on the session data", "priority": "high|medium|low", "content_hint": "brief description"}}
  ],
  "memory_proposals": [
    {{"action": "add|replace", "target": "memory|user", "content": "the exact text to add/replace", "reason": "why this should be remembered based on the evidence"}}
  ],
  "cross_agent_observations": [
    {{"observation": "what was learned", "relevant_agents": ["agent1", "agent2"], "reason": "why they should know this"}}
  ],
  "skill_gap_queries": ["search terms for skill hub"],
  "overall_confidence": 0.0
}}

RULES:
- Only propose improvements with CLEAR EVIDENCE from the session data above.
- Prefer skill proposals for REPEATED patterns (3+ similar sessions).
- Prefer memory entries for one-off learnings or behavioral notes.
- Cross-agent observations should be genuinely useful to the named agents based on their roles.
- Set confidence below 0.5 if the data is too sparse to draw conclusions.
- If nothing needs improvement, return empty arrays and confidence 0.0.
- Memory content must be concise (under 200 chars per entry).
"""


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_reflection_json(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*\n?", "", cleaned)
        cleaned = cleaned.rstrip("`").rstrip()
    try:
        return json.loads(cleaned)
    except Exception:
        pass
    m = _JSON_RE.search(text)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


async def _call_haiku(prompt: str) -> Optional[str]:
    from agent.auxiliary_client import async_call_llm
    try:
        resp = await async_call_llm(
            task="reflection",
            provider="anthropic",
            model=HAIKU_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=2000,
        )
        choice = resp.choices[0]
        return getattr(choice.message, "content", None) or ""
    except Exception as exc:
        logger.warning("Haiku reflection call failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------


def _apply_memory_add(agent_name: str, content: str) -> None:
    memory_path = _agent_dir(agent_name) / "MEMORY.md"
    memory_path.parent.mkdir(parents=True, exist_ok=True)
    tagged = f"- {AUTO_REFLECTION_TAG} {content.strip()}"
    existing = ""
    if memory_path.is_file():
        existing = memory_path.read_text(encoding="utf-8")
    if tagged in existing:
        return
    sep = "" if existing.endswith("\n") or not existing else "\n"
    with open(memory_path, "a", encoding="utf-8") as f:
        if not existing:
            f.write("# MEMORY\n\n")
        f.write(f"{sep}{tagged}\n")


def _apply_memory_replace(agent_name: str, old: str, new: str) -> Optional[str]:
    memory_path = _agent_dir(agent_name) / "MEMORY.md"
    if not memory_path.is_file():
        return None
    existing = memory_path.read_text(encoding="utf-8")
    if old not in existing:
        return None
    tagged_new = f"{AUTO_REFLECTION_TAG} {new.strip()}"
    memory_path.write_text(existing.replace(old, tagged_new, 1), encoding="utf-8")
    return old


async def _capture_cross_agent_observation(
    agent_name: str, obs: Dict[str, Any]
) -> bool:
    try:
        from hermes_cli.agent_registry import get_agent
        from tools.persistent_delegate_tool import PersistentDelegateTool
        profile = get_agent(agent_name)
        tool = PersistentDelegateTool()
        provider = tool._make_provider(profile)
        try:
            observation = obs.get("observation", "")
            relevant = ", ".join(obs.get("relevant_agents") or [])
            reason = obs.get("reason", "")
            text = (
                f"REFLECTION by {agent_name}: {observation}\n"
                f"Relevant agents: {relevant}\n"
                f"Reason: {reason}"
            )
            await provider.capture(text, source="reflection")
            return True
        finally:
            await provider.aclose()
    except Exception as exc:
        logger.warning("cross-agent capture failed: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


async def run_reflection(
    agent_name: str,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    dry_run: bool = False,
) -> ReflectionOutput:
    """Run a reflection cycle for *agent_name*.

    Returns a :class:`ReflectionOutput` even on failure (empty fields).
    """
    out = ReflectionOutput()
    # Gather — synchronous; avoid re-entering the running loop for compile.
    rin = ReflectionInput(
        agent_name=agent_name,
        recent_sessions=[],
        lookback_days=lookback_days,
    )
    sessions = _query_sessions(agent_name, lookback_days)
    rin.recent_sessions = sessions["all"]
    rin.positive_sessions = sessions["positive"]
    rin.negative_sessions = sessions["negative"]
    rin.tool_usage = _query_tool_usage(agent_name, lookback_days)
    rin.current_skills = _list_skills(agent_name)
    rin.memory_snapshot = _read_text_file(_agent_dir(agent_name) / "MEMORY.md", 4000)
    rin.user_snapshot = _read_text_file(_agent_dir(agent_name) / "USER.md", 2000)
    rin.compiled_context = await _try_compile_context(agent_name)

    if len(rin.recent_sessions) < MIN_SESSIONS_REQUIRED:
        _append_log(agent_name, {
            "action": "skipped",
            "reason": "insufficient_sessions",
            "data": {"session_count": len(rin.recent_sessions)},
            "applied": False,
        })
        return out

    prompt = _build_reflection_prompt(rin)
    raw = await _call_haiku(prompt)
    if not raw:
        _append_log(agent_name, {
            "action": "skipped",
            "reason": "llm_call_failed",
            "applied": False,
        })
        return out
    parsed = _parse_reflection_json(raw)
    if not parsed:
        _append_log(agent_name, {
            "action": "skipped",
            "reason": "parse_error",
            "data": {"raw": raw[:500]},
            "applied": False,
        })
        return out

    out.skill_proposals = parsed.get("skill_proposals") or []
    out.memory_proposals = parsed.get("memory_proposals") or []
    out.cross_agent_observations = parsed.get("cross_agent_observations") or []
    out.skill_gap_queries = parsed.get("skill_gap_queries") or []
    try:
        out.overall_confidence = float(parsed.get("overall_confidence") or 0.0)
    except (TypeError, ValueError):
        out.overall_confidence = 0.0

    _append_log(agent_name, {
        "action": "reflection_complete",
        "data": {
            "overall_confidence": out.overall_confidence,
            "memory_count": len(out.memory_proposals),
            "skill_count": len(out.skill_proposals),
            "cross_agent_count": len(out.cross_agent_observations),
            "dry_run": dry_run,
        },
        "applied": False,
    })

    if dry_run or out.overall_confidence < MIN_OVERALL_CONFIDENCE:
        if not dry_run:
            _append_log(agent_name, {
                "action": "skipped",
                "reason": "below_overall_confidence",
                "data": {"confidence": out.overall_confidence},
                "applied": False,
            })
        return out

    # Apply memory proposals (capped)
    applied_memory = 0
    for mp in out.memory_proposals:
        if applied_memory >= MAX_MEMORY_PER_CYCLE:
            _append_log(agent_name, {
                "action": "skipped",
                "reason": "memory_cap_reached",
                "data": mp,
                "applied": False,
            })
            continue
        action = (mp.get("action") or "add").lower()
        target = (mp.get("target") or "memory").lower()
        if target != "memory":
            _append_log(agent_name, {
                "action": "skipped",
                "reason": "user_md_edits_disabled",
                "data": mp,
                "applied": False,
            })
            continue
        threshold = CONFIDENCE_THRESHOLDS.get(
            f"memory_{action}", CONFIDENCE_THRESHOLDS["memory_add"]
        )
        if out.overall_confidence < threshold:
            _append_log(agent_name, {
                "action": "skipped_below_threshold",
                "data": {"proposal": mp, "threshold": threshold,
                         "confidence": out.overall_confidence},
                "applied": False,
            })
            continue
        content = (mp.get("content") or "").strip()
        if not content:
            continue
        try:
            if action == "replace":
                old_target = (mp.get("target_text") or content)
                old_value = _apply_memory_replace(agent_name, old_target, content)
                _append_log(agent_name, {
                    "action": "memory_replace",
                    "data": {"content": content, "replaced": old_value,
                             "reason": mp.get("reason")},
                    "applied": old_value is not None,
                })
                if old_value is not None:
                    applied_memory += 1
            else:
                _apply_memory_add(agent_name, content)
                _append_log(agent_name, {
                    "action": "memory_add",
                    "data": {"content": f"{AUTO_REFLECTION_TAG} {content}",
                             "reason": mp.get("reason")},
                    "applied": True,
                })
                applied_memory += 1
        except Exception as exc:
            logger.warning("memory apply failed: %s", exc)
            _append_log(agent_name, {
                "action": "error",
                "data": {"proposal": mp, "error": str(exc)},
                "applied": False,
            })

    # Skill proposals — log only, never auto-apply
    for sp in out.skill_proposals:
        _append_log(agent_name, {
            "action": "skill_proposal",
            "data": sp,
            "applied": False,
            "reason": "requires_user_review",
        })

    # Cross-agent observations
    if out.overall_confidence >= CONFIDENCE_THRESHOLDS["cross_agent"]:
        for obs in out.cross_agent_observations:
            ok = await _capture_cross_agent_observation(agent_name, obs)
            _append_log(agent_name, {
                "action": "cross_agent_capture",
                "data": obs,
                "applied": ok,
            })

    # Skill gap queries
    for q in out.skill_gap_queries:
        _append_log(agent_name, {
            "action": "skill_gap_search",
            "data": {"query": q},
            "applied": False,
        })

    return out


# ---------------------------------------------------------------------------
# Cron registration
# ---------------------------------------------------------------------------


def stagger_schedule(agent_name: str) -> str:
    """Generate a staggered daily cron schedule based on agent name hash."""
    h = int(hashlib.md5(agent_name.encode()).hexdigest()[:8], 16)
    hour = (h % 6) + 2   # 2am - 7am
    minute = h % 60
    return f"{minute} {hour} * * *"


def _reflection_job_name(agent_name: str) -> str:
    return f"reflection:{agent_name}"


def ensure_reflection_cron_jobs(
    agents: Optional[List[str]] = None,
) -> Dict[str, str]:
    """Register a daily reflection cron job for each agent.

    Idempotent: jobs with the reserved name prefix ``reflection:<agent>``
    are replaced if already present so the schedule stays in sync with
    :func:`stagger_schedule`.

    Returns a mapping of ``agent_name -> schedule``.
    """
    from cron.jobs import load_jobs, save_jobs, create_job, remove_job
    from hermes_cli.agent_registry import list_agents

    names = agents or list_agents()
    schedules: Dict[str, str] = {}
    existing_jobs = load_jobs()
    existing_by_name = {j.get("name"): j for j in existing_jobs}

    for agent in names:
        job_name = _reflection_job_name(agent)
        schedule = stagger_schedule(agent)
        schedules[agent] = schedule
        if job_name in existing_by_name:
            prior = existing_by_name[job_name]
            if prior.get("schedule_display") == schedule and prior.get("job_type") == "reflection":
                continue
            try:
                remove_job(prior["id"])
            except Exception:
                pass

        prompt = (
            f"[reflection job for {agent}] "
            "This job is handled by the reflection runner, not the agent."
        )
        job = create_job(
            prompt=prompt,
            schedule=schedule,
            name=job_name,
            repeat=None,
            deliver="local",
        )
        # Tag the job so the scheduler can dispatch it to run_reflection.
        jobs = load_jobs()
        for j in jobs:
            if j["id"] == job["id"]:
                j["job_type"] = "reflection"
                j["reflection_agent"] = agent
                break
        save_jobs(jobs)

    return schedules


def run_reflection_job(job: Dict[str, Any]) -> Dict[str, Any]:
    """Synchronous entry point used by the cron scheduler."""
    agent = job.get("reflection_agent") or ""
    if not agent:
        return {"ok": False, "error": "missing reflection_agent"}
    try:
        result = asyncio.run(run_reflection(agent, dry_run=False))
        return {
            "ok": True,
            "confidence": result.overall_confidence,
            "memory_proposals": len(result.memory_proposals),
            "skill_proposals": len(result.skill_proposals),
            "cross_agent_observations": len(result.cross_agent_observations),
        }
    except Exception as exc:  # pragma: no cover
        logger.exception("reflection job failed: %s", exc)
        return {"ok": False, "error": str(exc)}
