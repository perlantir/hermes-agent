"""An in-process mock HIPP0 server for hermes-agent tests.

Implements the six locked endpoints from the ``feat/persistent-agents-hipp0``
task brief:

    POST /api/hermes/session/start
    POST /api/hermes/session/end
    POST /api/hermes/register
    POST /api/capture
    POST /api/compile
    POST /api/outcomes
    POST /api/hermes/user-facts

The mock exposes:

- :func:`start_mock_hipp0` — async context manager that yields a
  :class:`MockHipp0` object (``base_url``, ``calls``, handle to inject
  failures).
- :class:`MockHipp0` — lets tests assert against the recorded call log,
  inject 5xx failures, simulate outages, and stub compile responses.

It runs a real aiohttp TCP server on ``127.0.0.1:<random>`` so the
code under test exercises the actual httpx → TCP → aiohttp path.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Optional

from aiohttp import web


@dataclass
class MockCall:
    """One recorded HTTP call against the mock."""

    method: str
    path: str
    query: Dict[str, str]
    body: Dict[str, Any]
    headers: Dict[str, str]


@dataclass
class MockHipp0:
    """Handle returned by :func:`start_mock_hipp0`."""

    base_url: str
    runner: web.AppRunner
    calls: List[MockCall] = field(default_factory=list)

    # Failure injection knobs — tests mutate these to simulate HIPP0
    # outages, flakiness, 4xx contract bugs, etc.
    force_status_for_path: Dict[str, int] = field(default_factory=dict)
    force_status_remaining: Dict[str, int] = field(default_factory=dict)
    compile_response: Optional[Dict[str, Any]] = None
    capture_response: Optional[Dict[str, Any]] = None
    register_response: Optional[Dict[str, Any]] = None
    session_start_response: Optional[Dict[str, Any]] = None

    def last_call(self, path: Optional[str] = None) -> Optional[MockCall]:
        if path is None:
            return self.calls[-1] if self.calls else None
        for c in reversed(self.calls):
            if c.path == path:
                return c
        return None

    def calls_for(self, path: str) -> List[MockCall]:
        return [c for c in self.calls if c.path == path]

    def queue_failure(self, path: str, status: int, count: int = 1) -> None:
        """Make the next ``count`` requests to ``path`` return ``status``."""
        self.force_status_for_path[path] = status
        self.force_status_remaining[path] = count

    async def aclose(self) -> None:
        await self.runner.cleanup()


def _extract_failure(
    state: MockHipp0, path: str
) -> Optional[int]:
    remaining = state.force_status_remaining.get(path, 0)
    if remaining <= 0:
        return None
    status = state.force_status_for_path.get(path)
    state.force_status_remaining[path] = remaining - 1
    if state.force_status_remaining[path] == 0:
        # Clear the knob so subsequent calls succeed.
        state.force_status_remaining.pop(path, None)
        state.force_status_for_path.pop(path, None)
    return status


def _build_app(state: MockHipp0) -> web.Application:
    app = web.Application()

    async def _record(request: web.Request) -> Dict[str, Any]:
        body_text = await request.text()
        body: Dict[str, Any] = {}
        if body_text:
            try:
                body = json.loads(body_text)
            except json.JSONDecodeError:
                body = {"_raw": body_text}
        call = MockCall(
            method=request.method,
            path=request.path,
            query=dict(request.query),
            body=body,
            headers=dict(request.headers),
        )
        state.calls.append(call)
        return body

    async def session_start(request: web.Request) -> web.Response:
        await _record(request)
        if (s := _extract_failure(state, request.path)) is not None:
            return web.json_response({"error": "forced"}, status=s)
        payload = state.session_start_response or {
            "session_id": f"session-{uuid.uuid4()}"
        }
        return web.json_response(payload, status=201)

    async def session_end(request: web.Request) -> web.Response:
        await _record(request)
        if (s := _extract_failure(state, request.path)) is not None:
            return web.json_response({"error": "forced"}, status=s)
        return web.json_response(
            {"summary_snippet_ids": [f"snip-{uuid.uuid4()}"]}
        )

    async def register(request: web.Request) -> web.Response:
        body = await _record(request)
        if (s := _extract_failure(state, request.path)) is not None:
            return web.json_response({"error": "forced"}, status=s)
        payload = state.register_response or {
            "agent_id": f"agent-{body.get('agent_name', 'x')}-uuid",
            "created": True,
        }
        return web.json_response(payload, status=201)

    async def capture(request: web.Request) -> web.Response:
        await _record(request)
        if (s := _extract_failure(state, request.path)) is not None:
            return web.json_response({"error": "forced"}, status=s)
        payload = state.capture_response or {
            "capture_id": f"cap-{uuid.uuid4()}",
            "status": "processing",
        }
        return web.json_response(payload, status=202)

    async def compile_(request: web.Request) -> web.Response:
        await _record(request)
        if (s := _extract_failure(state, request.path)) is not None:
            return web.json_response({"error": "forced"}, status=s)
        payload = state.compile_response or {
            "decisions": [
                {
                    "id": "dec-1",
                    "text": "Test decision from mock HIPP0",
                    "score": 0.9,
                    "source_session": "session-mock",
                }
            ],
            "total_tokens": 42,
            "cache_hit": False,
            "role_signal": None,
            "contrastive_pairs": None,
        }
        return web.json_response(payload, status=200)

    async def outcomes(request: web.Request) -> web.Response:
        await _record(request)
        if (s := _extract_failure(state, request.path)) is not None:
            return web.json_response({"error": "forced"}, status=s)
        return web.json_response(
            {"recorded": True, "weight_updates_scheduled": True}
        )

    async def user_facts(request: web.Request) -> web.Response:
        await _record(request)
        if (s := _extract_failure(state, request.path)) is not None:
            return web.json_response({"error": "forced"}, status=s)
        return web.json_response(
            {
                "version": f"etag-{uuid.uuid4()}",
                "facts": [],
            }
        )

    app.router.add_post("/api/hermes/session/start", session_start)
    app.router.add_post("/api/hermes/session/end", session_end)
    app.router.add_post("/api/hermes/register", register)
    app.router.add_post("/api/capture", capture)
    app.router.add_post("/api/compile", compile_)
    app.router.add_post("/api/outcomes", outcomes)
    app.router.add_post("/api/hermes/user-facts", user_facts)
    return app


@asynccontextmanager
async def start_mock_hipp0(
    host: str = "127.0.0.1",
) -> AsyncIterator[MockHipp0]:
    """Start an in-process mock HIPP0 server.

    Yields a :class:`MockHipp0` whose ``base_url`` can be passed to
    :class:`agent.hipp0_memory_provider.Hipp0MemoryProvider`.

    On exit the server is torn down via ``AppRunner.cleanup``.
    """
    # Placeholder so the closure captured by _build_app has a state
    # object from the start — we fill in runner + base_url after bind.
    state = MockHipp0(base_url="", runner=None)  # type: ignore[arg-type]
    app = _build_app(state)

    runner = web.AppRunner(app)
    await runner.setup()

    site = web.TCPSite(runner, host=host, port=0)
    await site.start()

    # aiohttp picks a free port; introspect it.
    server = site._server  # noqa: SLF001 — intentional
    assert server is not None
    assert server.sockets, "aiohttp site has no bound sockets"
    bound_port = server.sockets[0].getsockname()[1]

    state.base_url = f"http://{host}:{bound_port}"
    state.runner = runner
    try:
        yield state
    finally:
        await runner.cleanup()


# ---------------------------------------------------------------------------
# pytest fixture
# ---------------------------------------------------------------------------


try:  # pragma: no cover - fixture plumbing
    import pytest_asyncio

    @pytest_asyncio.fixture
    async def mock_hipp0() -> AsyncIterator[MockHipp0]:
        async with start_mock_hipp0() as state:
            yield state

except ImportError:  # pragma: no cover - optional dep
    pass
