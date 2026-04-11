"""HIPP0-backed memory provider for persistent Hermes agents.

This is the HTTP client that sits between a named persistent agent
(``hermes_cli.agent_registry``) and a running HIPP0 instance. It talks
the contract locked down in the ``feat/persistent-agents-hipp0`` task
brief — see ``HIPP0_REQUESTS.md`` at the repo root for any divergences
this file has had to flag while the sibling repo evolves.

The class implements the sync :class:`agent.memory_provider.MemoryProvider`
abstract base so Hermes runtime plumbing can hold it, but its *primary*
API is the set of ``async`` HTTP methods consumed directly by
:class:`tools.persistent_delegate_tool.PersistentDelegateTool`:

    - :meth:`start_session` / :meth:`end_session`
    - :meth:`capture` — fire-and-forget conversation ingestion
    - :meth:`compile` — task-aware context retrieval
    - :meth:`record_outcome` — reinforcement signal
    - :meth:`upsert_user_fact` — cross-agent user facts

Write-Ahead Log
---------------
Every async call is wrapped in a WAL. A failed request (connection
error or 5xx) is appended as one JSON line to the agent's
``pending.jsonl``. On the next successful call we drain that WAL
oldest-first, dropping lines as they replay. This is the mechanism
that prevents memory loss when HIPP0 is transiently down.

Degraded mode
-------------
If :meth:`compile` fails after retries we fall back to reading
``MEMORY.md`` off disk and return a :class:`CompiledContext` with
``degraded=True``. Callers (PersistentDelegateTool, gateway) should
surface the flag in the agent's first line so users know why recall
looks thin.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import httpx

from agent.memory_provider import MemoryProvider

logger = logging.getLogger(__name__)


# Outcome literal — matches HIPP0's /api/outcomes contract.
OutcomeLiteral = Literal["positive", "negative", "neutral"]

# A conservative per-request timeout. HIPP0 is expected to be local or
# in-region; long tails usually mean a dead peer we should fall back
# from rather than wait on.
_DEFAULT_TIMEOUT_SECONDS = 15.0
_DEFAULT_CONNECT_TIMEOUT_SECONDS = 5.0

# How many times we retry an individual HTTP call (connection or 5xx)
# before writing the request to the WAL.
_RETRY_ATTEMPTS = 3
_RETRY_INITIAL_DELAY = 0.4  # seconds; doubles each retry


# ---------------------------------------------------------------------------
# Response dataclasses
# ---------------------------------------------------------------------------


@dataclass
class CompiledContext:
    """Result of a ``POST /api/compile`` call.

    ``degraded=True`` means HIPP0 was unreachable and the payload was
    reconstructed from the agent's local ``MEMORY.md`` snapshot.
    """

    decisions: List[Dict[str, Any]] = field(default_factory=list)
    total_tokens: int = 0
    cache_hit: bool = False
    role_signal: Optional[Dict[str, Any]] = None
    contrastive_pairs: Optional[List[Dict[str, Any]]] = None
    degraded: bool = False
    degraded_reason: Optional[str] = None

    def as_prompt_block(self) -> str:
        """Render the compiled context as a plain-text prompt block.

        Used by PersistentDelegateTool when constructing the delegate's
        system prompt. Keeps a deterministic shape so snapshot tests are
        stable.
        """
        if self.degraded:
            header = "## Compiled context (DEGRADED — HIPP0 unreachable)"
            if self.degraded_reason:
                header += f"\n_Reason: {self.degraded_reason}_"
        else:
            header = "## Compiled context"

        lines = [header, ""]
        if not self.decisions:
            lines.append("_(no decisions returned)_")
            return "\n".join(lines)
        for d in self.decisions:
            text = d.get("text", "")
            score = d.get("score")
            did = d.get("id", "?")
            lines.append(f"- [{did}] {text}" + (f" (score={score})" if score is not None else ""))
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class Hipp0Error(RuntimeError):
    """Base class for HIPP0 HTTP failures."""


class Hipp0HTTPError(Hipp0Error):
    """Raised when HIPP0 returned a non-success status the caller can see."""

    def __init__(self, status_code: int, body: str):
        super().__init__(f"HIPP0 {status_code}: {body[:200]}")
        self.status_code = status_code
        self.body = body


class Hipp0UnavailableError(Hipp0Error):
    """Raised when HIPP0 is unreachable after retries (conn error or 5xx)."""


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------


class Hipp0MemoryProvider(MemoryProvider):
    """HTTP memory provider backed by a HIPP0 instance.

    Instances are bound to a specific agent (``agent_name`` / ``agent_id``)
    and a specific HIPP0 project (``project_id``). One instance owns one
    ``httpx.AsyncClient`` with connection pooling; reuse it across calls.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        project_id: str,
        agent_name: str,
        agent_id: str,
        *,
        pending_wal_path: Optional[Path] = None,
        memory_md_path: Optional[Path] = None,
        client: Optional[httpx.AsyncClient] = None,
        timeout: float = _DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.project_id = project_id
        self.agent_name = agent_name
        self.agent_id = agent_id

        self._pending_wal_path = Path(pending_wal_path) if pending_wal_path else None
        self._memory_md_path = Path(memory_md_path) if memory_md_path else None

        self._session_id: Optional[str] = None

        self._client = client or httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(
                timeout,
                connect=_DEFAULT_CONNECT_TIMEOUT_SECONDS,
            ),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "User-Agent": f"hermes-agent/hipp0-provider ({agent_name})",
            },
        )
        self._owns_client = client is None

    # ------------------------------------------------------------------ core

    @property
    def name(self) -> str:  # pragma: no cover — trivial
        return "hipp0"

    @property
    def session_id(self) -> Optional[str]:
        return self._session_id

    def is_available(self) -> bool:  # pragma: no cover — config-only check
        return bool(self.base_url and self.api_key and self.project_id)

    def initialize(self, session_id: str, **kwargs: Any) -> None:
        """Sync stub to satisfy the :class:`MemoryProvider` ABC.

        The Hipp0 provider is driven primarily by the async API that
        :class:`tools.persistent_delegate_tool.PersistentDelegateTool`
        consumes directly. When Hermes's sync memory manager hosts the
        provider as a normal plugin, the session id has already been
        negotiated by the delegate tool — we just cache it so any
        subsequent capture/compile calls use the correct id.
        """
        if session_id:
            self._session_id = session_id

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        # This provider's value is in compile/capture orchestration
        # driven by PersistentDelegateTool, not in per-turn tool calls.
        return []

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    # ----------------------------------------------------------------- HTTP

    async def start_session(
        self,
        *,
        platform: str,
        user_id: Optional[str] = None,
        external_chat_id: Optional[str] = None,
    ) -> str:
        """Ask HIPP0 for a server-generated session id.

        HIPP0 owns all session ids — we never client-generate them. The
        returned id is cached on the provider and reused for subsequent
        capture/compile calls until :meth:`end_session` or a new
        :meth:`start_session`.
        """
        payload = {
            "project_id": self.project_id,
            "agent_name": self.agent_name,
            "platform": platform,
            "user_id": user_id,
            "external_chat_id": external_chat_id,
        }
        data = await self._post_json("/api/hermes/session/start", payload)
        session_id = data.get("session_id")
        if not session_id:
            raise Hipp0Error(f"session/start missing session_id: {data!r}")
        self._session_id = session_id
        return session_id

    async def end_session(self) -> List[str]:
        """End the current HIPP0 session and return any summary snippet ids."""
        if not self._session_id:
            return []
        payload = {"session_id": self._session_id}
        try:
            data = await self._post_json("/api/hermes/session/end", payload)
        finally:
            self._session_id = None
        return list(data.get("summary_snippet_ids") or [])

    async def capture(
        self,
        conversation_text: str,
        *,
        source: str = "hermes",
        source_event_id: Optional[str] = None,
        source_channel: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Fire-and-forget conversation ingestion.

        HIPP0 returns ``202`` with ``capture_id`` + ``status`` immediately;
        snippet extraction is async. Poll ``GET /api/capture/:id`` later
        if you need the extracted decision ids.
        """
        if len(conversation_text) > 500_000:
            raise ValueError("conversation_text exceeds 500_000 chars")
        payload = {
            "agent_name": self.agent_name,
            "project_id": self.project_id,
            "conversation": conversation_text,
            "session_id": self._session_id,
            "source": source,
            "source_event_id": source_event_id,
            "source_channel": source_channel,
        }
        return await self._post_json("/api/capture", payload, wal_kind="capture")

    async def compile(
        self,
        task_description: str,
        *,
        max_tokens: int = 4000,
        fast_mode: bool = True,
        task_session_id: Optional[str] = None,
        namespace: Optional[str] = None,
        session_lookback_days: Optional[int] = None,
        include_superseded: bool = False,
        include_role_signal: bool = True,
    ) -> CompiledContext:
        """Task-aware context retrieval with degraded-mode fallback."""
        if len(task_description) > 100_000:
            raise ValueError("task_description exceeds 100_000 chars")

        body: Dict[str, Any] = {
            "agent_name": self.agent_name,
            "project_id": self.project_id,
            "task_description": task_description,
            "max_tokens": max_tokens,
            "include_superseded": include_superseded,
            "include_role_signal": include_role_signal,
        }
        if task_session_id is not None:
            body["task_session_id"] = task_session_id
        if namespace is not None:
            body["namespace"] = namespace
        if session_lookback_days is not None:
            body["session_lookback_days"] = session_lookback_days

        # Fast mode: skip shared-pattern expansion and explanations,
        # tighten the score threshold. Full mode is reserved for
        # session-start compiles.
        if fast_mode:
            params = {
                "format": "json",
                "depth": "default",
                "threshold": "0.6",
                "include_patterns": "false",
                "explain": "false",
            }
        else:
            params = {
                "format": "json",
                "depth": "full",
                "threshold": "0.5",
                "include_patterns": "true",
                "explain": "false",
            }

        try:
            data = await self._post_json(
                "/api/compile",
                body,
                params=params,
                wal_kind="compile",
                allow_wal=False,  # compile is read; no point queueing
            )
        except Hipp0UnavailableError as e:
            return self._degraded_compile(str(e))
        except Hipp0HTTPError as e:
            # 4xx is a hard contract bug — surface it. 5xx fell through
            # to Hipp0UnavailableError via retry.
            if 500 <= e.status_code < 600:
                return self._degraded_compile(str(e))
            raise

        return CompiledContext(
            decisions=list(data.get("decisions") or []),
            total_tokens=int(data.get("total_tokens") or 0),
            cache_hit=bool(data.get("cache_hit")),
            role_signal=data.get("role_signal"),
            contrastive_pairs=data.get("contrastive_pairs"),
            degraded=False,
        )

    async def record_outcome(
        self,
        snippet_ids: List[str],
        outcome: OutcomeLiteral,
        *,
        signal_source: str,
    ) -> None:
        """Record reinforcement signal for a set of snippets.

        Hermes must call this at end of turn. ``signal_source`` must be
        one of ``user_reaction``, ``auto_detect``, ``explicit_feedback``
        per the HIPP0 contract.
        """
        if not snippet_ids:
            return
        payload = {
            "project_id": self.project_id,
            "session_id": self._session_id,
            "agent_name": self.agent_name,
            "snippet_ids": list(snippet_ids),
            "outcome": outcome,
            "signal_source": signal_source,
        }
        await self._post_json("/api/outcomes", payload, wal_kind="outcome")

    async def upsert_user_fact(
        self,
        user_id: str,
        facts: List[Dict[str, Any]],
        *,
        etag: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Upsert facts about a user with optimistic concurrency control.

        ``facts`` is a list of ``{"key", "value", "additive"}`` dicts.
        Pass the ``etag`` returned by the last successful upsert as the
        ``If-Match`` header to avoid clobbering concurrent edits.
        """
        payload = {
            "project_id": self.project_id,
            "user_id": user_id,
            "facts": list(facts),
        }
        headers = {"If-Match": etag} if etag else None
        return await self._post_json(
            "/api/hermes/user-facts",
            payload,
            extra_headers=headers,
        )

    async def register(self, soul: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Register this agent with HIPP0 and return ``{agent_id, created}``.

        AgentRegistry.register_agent() writes the profile locally; callers
        that want a HIPP0 agent id should invoke this afterwards and then
        persist ``agent_id`` into the local config via
        :func:`hermes_cli.agent_registry.update_agent_config`.
        """
        payload = {
            "project_id": self.project_id,
            "agent_name": self.agent_name,
            "soul": soul,
            "config": config,
        }
        data = await self._post_json("/api/hermes/register", payload)
        returned_id = data.get("agent_id")
        if returned_id:
            self.agent_id = returned_id
        return data

    # --------------------------------------------------------- HTTP internals

    async def _post_json(
        self,
        path: str,
        body: Dict[str, Any],
        *,
        params: Optional[Dict[str, str]] = None,
        extra_headers: Optional[Dict[str, str]] = None,
        wal_kind: Optional[str] = None,
        allow_wal: bool = True,
    ) -> Dict[str, Any]:
        """POST JSON with retries, WAL on outage, and WAL drain on success."""
        attempt = 0
        last_exc: Optional[BaseException] = None
        while attempt < _RETRY_ATTEMPTS:
            try:
                resp = await self._client.post(
                    path,
                    json=body,
                    params=params,
                    headers=extra_headers,
                )
            except (httpx.TransportError, httpx.TimeoutException) as e:
                last_exc = e
                await asyncio.sleep(_RETRY_INITIAL_DELAY * (2**attempt))
                attempt += 1
                continue

            if 500 <= resp.status_code < 600:
                last_exc = Hipp0HTTPError(resp.status_code, resp.text)
                await asyncio.sleep(_RETRY_INITIAL_DELAY * (2**attempt))
                attempt += 1
                continue

            if resp.status_code >= 400:
                # 4xx: hard client-side contract error. Don't WAL, don't retry.
                raise Hipp0HTTPError(resp.status_code, resp.text)

            # Success — drain any queued WAL entries in the background
            # (best-effort). If draining fails the WAL stays put.
            if wal_kind is not None:
                await self._drain_wal()
            try:
                return resp.json() if resp.content else {}
            except json.JSONDecodeError:
                return {}

        # Exhausted retries.
        if wal_kind is not None and allow_wal:
            self._wal_append(
                {
                    "kind": wal_kind,
                    "path": path,
                    "body": body,
                    "params": params,
                    "headers": extra_headers,
                    "timestamp": time.time(),
                    "error": str(last_exc),
                }
            )
            logger.warning(
                "HIPP0 %s unreachable after %d retries; queued to WAL: %s",
                path,
                _RETRY_ATTEMPTS,
                last_exc,
            )
        raise Hipp0UnavailableError(
            f"HIPP0 {path} unreachable after {_RETRY_ATTEMPTS} retries: {last_exc}"
        )

    # ----------------------------------------------------------------- WAL

    def _wal_append(self, record: Dict[str, Any]) -> None:
        if not self._pending_wal_path:
            logger.error(
                "HIPP0 WAL miss: no pending.jsonl path configured; dropping %s",
                record.get("kind"),
            )
            return
        self._pending_wal_path.parent.mkdir(parents=True, exist_ok=True)
        with self._pending_wal_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    async def _drain_wal(self) -> None:
        """Replay WAL entries oldest-first. Drops on success, keeps on failure."""
        if not self._pending_wal_path or not self._pending_wal_path.exists():
            return
        try:
            lines = self._pending_wal_path.read_text(encoding="utf-8").splitlines()
        except OSError as e:
            logger.warning("HIPP0 WAL: could not read %s: %s", self._pending_wal_path, e)
            return

        remaining: List[str] = []
        for i, line in enumerate(lines):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("HIPP0 WAL: dropping malformed line %d", i)
                continue
            try:
                resp = await self._client.post(
                    record["path"],
                    json=record.get("body") or {},
                    params=record.get("params"),
                    headers=record.get("headers"),
                )
            except (httpx.TransportError, httpx.TimeoutException):
                # Keep this line and all subsequent lines in order.
                remaining.append(line)
                remaining.extend(lines[i + 1 :])
                break
            if resp.status_code >= 500:
                remaining.append(line)
                remaining.extend(lines[i + 1 :])
                break
            # 4xx or 2xx: drop the entry (4xx means bad contract; there's
            # no point retrying a malformed request forever).
            if resp.status_code >= 400:
                logger.warning(
                    "HIPP0 WAL: dropping 4xx entry %s on drain (%d)",
                    record.get("kind"),
                    resp.status_code,
                )
            continue

        if remaining:
            self._pending_wal_path.write_text(
                "\n".join(remaining) + "\n", encoding="utf-8"
            )
        else:
            try:
                self._pending_wal_path.unlink()
            except OSError:
                pass

    def wal_size(self) -> int:
        """Return the number of queued WAL entries (test + observability helper)."""
        if not self._pending_wal_path or not self._pending_wal_path.exists():
            return 0
        try:
            return sum(
                1 for line in self._pending_wal_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
        except OSError:
            return 0

    # -------------------------------------------------------- degraded mode

    def _degraded_compile(self, reason: str) -> CompiledContext:
        """Build a CompiledContext from the local MEMORY.md cache."""
        decisions: List[Dict[str, Any]] = []
        total_tokens = 0
        if self._memory_md_path and self._memory_md_path.is_file():
            try:
                text = self._memory_md_path.read_text(encoding="utf-8")
            except OSError as e:
                logger.warning(
                    "HIPP0 degraded: failed to read %s: %s",
                    self._memory_md_path,
                    e,
                )
                text = ""
            if text.strip():
                decisions.append(
                    {
                        "id": "local-memory",
                        "text": text,
                        "score": None,
                        "source": "MEMORY.md",
                    }
                )
                # Rough token estimate: 4 chars per token.
                total_tokens = max(1, len(text) // 4)
        logger.warning("HIPP0 compile degraded: %s", reason)
        return CompiledContext(
            decisions=decisions,
            total_tokens=total_tokens,
            cache_hit=False,
            degraded=True,
            degraded_reason=reason,
        )
