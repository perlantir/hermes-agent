"""WebSocket server for the HIPP0 dashboard Chat view.

Listens on WEB_PLATFORM_PORT (default 3300) and speaks the Chat
protocol defined in the HIPP0 parallel-build spec:

Client -> Server:
  { "type": "message", "agent_name": "alice", "content": "...", "conversation_id": "..." }
  { "type": "command", "command": "new|stop", "agent_name": "alice" }

Server -> Client:
  { "type": "stream_start", "conversation_id": "...", "agent_name": "...", "model": "..." }
  { "type": "stream_delta", "content": "partial token" }
  { "type": "tool_call", "tool_name": "...", "tool_emoji": "...", "status": "...", ... }
  { "type": "stream_end", "conversation_id": "...", "tokens": {...}, "duration_seconds": ... }
  { "type": "error", "message": "...", "recoverable": true }

Each WebSocket connection maintains its own AIAgent instances keyed by
agent_name. The HIPP0 memory provider (compile before turn, capture
after turn) is wired identically to hermes_cli/repl.py.

Usage:
    python -m gateway.platforms.web_platform
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import websockets
import websockets.server

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Ensure repo root is on sys.path so hermes imports resolve
# ---------------------------------------------------------------------------

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from hermes_cli.agent_registry import AgentNotFoundError, get_agent, list_agents
from agent.hipp0_memory_provider import Hipp0MemoryProvider, CompiledContext
from agent.prompt_builder import build_slim_system_prompt

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

WEB_PLATFORM_PORT = int(os.environ.get("WEB_PLATFORM_PORT", "3300"))
HIPP0_BASE_URL = os.environ.get("HIPP0_BASE_URL", "http://127.0.0.1:3100")
HIPP0_API_KEY_FILE = os.environ.get("HIPP0_API_KEY_FILE", "/etc/team-hippo/api-key.txt")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

# CORS origins allowed for WebSocket upgrades
ALLOWED_ORIGINS = {
    "https://app.hipp0.ai",
    "http://app.hipp0.ai",
    "http://localhost:3200",
    "http://127.0.0.1:3200",
}


# ---------------------------------------------------------------------------
# Sync adapter for HIPP0 memory (same pattern as repl.py)
# ---------------------------------------------------------------------------

class _Hipp0SyncAdapter:
    """Thin sync wrapper bridging async Hipp0MemoryProvider into the sync
    MemoryProvider ABC that AIAgent's MemoryManager expects."""

    def __init__(self, provider: Hipp0MemoryProvider, loop: asyncio.AbstractEventLoop) -> None:
        self._provider = provider
        self._loop = loop

    @property
    def name(self) -> str:
        return "hipp0"

    def is_available(self) -> bool:
        return self._provider.is_available()

    def initialize(self, session_id: str, **kwargs: Any) -> None:
        self._provider.initialize(session_id, **kwargs)

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return []

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        if not query:
            return ""
        try:
            compiled = self._loop.run_until_complete(
                self._provider.compile(query, fast_mode=True)
            )
            return compiled.as_prompt_block()
        except Exception as e:
            logger.warning("HIPP0 prefetch failed: %s", e)
            return ""

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        pass

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        transcript = f"USER: {user_content}\nASSISTANT: {assistant_content}"
        try:
            self._loop.run_until_complete(
                self._provider.capture(transcript, source="hermes")
            )
        except Exception as e:
            logger.warning("HIPP0 capture failed: %s", e)

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


# ---------------------------------------------------------------------------
# Per-connection agent session
# ---------------------------------------------------------------------------

class _AgentSession:
    """Holds the AIAgent instance + HIPP0 provider for one (connection, agent) pair."""

    def __init__(
        self,
        agent_name: str,
        hipp0_key: str,
        ws_loop: asyncio.AbstractEventLoop,
    ) -> None:
        self.agent_name = agent_name
        self.conversation_id = str(uuid.uuid4())
        self._hipp0_key = hipp0_key
        self._ws_loop = ws_loop
        self._agent = None
        self._provider = None
        self._agent_loop = None
        self._model = "claude-sonnet-4-6"
        self._interrupted = False

    def setup(self) -> None:
        """Initialize the AIAgent + HIPP0 provider (called in worker thread)."""
        from run_agent import AIAgent
        from agent.memory_manager import MemoryManager

        profile = get_agent(self.agent_name)
        if not profile.config.project_id:
            raise ValueError(f"Agent {self.agent_name!r} has no project_id")

        self._model = profile.config.model or "claude-sonnet-4-6"

        # Create a dedicated event loop for this agent's async HIPP0 calls
        self._agent_loop = asyncio.new_event_loop()

        self._provider = Hipp0MemoryProvider(
            base_url=HIPP0_BASE_URL,
            api_key=self._hipp0_key,
            project_id=str(profile.config.project_id),
            agent_name=profile.name,
            agent_id=str(profile.config.agent_id or ""),
            pending_wal_path=profile.pending_wal_path,
            memory_md_path=profile.memory_path,
        )

        # Start HIPP0 session
        try:
            session_id = self._agent_loop.run_until_complete(
                self._provider.start_session(platform="web")
            )
        except Exception as e:
            logger.warning("Failed to start HIPP0 session for %s: %s", self.agent_name, e)
            session_id = str(uuid.uuid4())

        # Compile initial context
        try:
            compiled = self._agent_loop.run_until_complete(
                self._provider.compile("General conversation", fast_mode=False)
            )
        except Exception:
            compiled = CompiledContext(degraded=True, degraded_reason="compile failed at session start")

        # Fetch supplementary context (decisions, captures, user_facts)
        extra_sections = self._fetch_extra_context(profile)

        system_prompt = build_slim_system_prompt(
            profile.soul,
            compiled_context_block=compiled.as_prompt_block(),
            platform_hint="web",
            extra_sections=extra_sections if extra_sections else None,
        )

        # Ensure hermes directories exist
        hermes_home = Path.home() / ".hermes"
        (hermes_home / "memories").mkdir(parents=True, exist_ok=True)

        from hermes_state import SessionDB
        session_db = SessionDB()

        self._agent = AIAgent(
            model=self._model,
            provider="anthropic",
            api_key=ANTHROPIC_API_KEY,
            max_iterations=int(profile.config.extra.get("max_iterations", 50)),
            quiet_mode=True,
            ephemeral_system_prompt=system_prompt,
            platform="web",
            skip_context_files=True,
            skip_memory=False,
            slim_prompt=True,
            session_db=session_db,
        )

        # Wire HIPP0 as external memory provider
        adapter = _Hipp0SyncAdapter(self._provider, self._agent_loop)
        if self._agent._memory_manager is None:
            self._agent._memory_manager = MemoryManager()
        self._agent._memory_manager.add_provider(adapter)
        adapter.initialize(session_id, platform="web", hermes_home=str(hermes_home))

    def _fetch_extra_context(self, profile: Any) -> List[str]:
        """Fetch decisions, captures, user_facts from HIPP0 (same as repl.py)."""
        import httpx
        sections = []
        project_id = str(profile.config.project_id)
        headers = {"Authorization": f"Bearer {self._hipp0_key}"}

        # Decisions fallback
        try:
            resp = httpx.get(
                f"{HIPP0_BASE_URL}/api/decisions",
                params={"project_id": project_id, "limit": "20"},
                headers=headers, timeout=10,
            )
            if resp.status_code == 200:
                decisions = resp.json()
                if decisions:
                    lines = ["## Project decisions (from HIPP0 memory)", ""]
                    for d in decisions:
                        title = d.get("title", "untitled")
                        desc = (d.get("description") or "")[:300]
                        by = d.get("made_by", "?")
                        lines.append(f"- **{title}** (by {by}): {desc}")
                    sections.append("\n".join(lines))
        except Exception:
            pass

        # Recent captures
        try:
            resp = httpx.get(
                f"{HIPP0_BASE_URL}/api/hermes/captures",
                params={"project_id": project_id, "agent_name": profile.name, "limit": "10"},
                headers=headers, timeout=10,
            )
            if resp.status_code == 200:
                captures = resp.json()
                if captures:
                    lines = ["## Recent conversations (from HIPP0 memory)", ""]
                    for cap in captures:
                        text = (cap.get("conversation_text") or "")[:500]
                        ts = cap.get("created_at", "unknown")
                        if text.strip():
                            lines.append(f"### Session ({ts})")
                            lines.append(text)
                            lines.append("")
                    sections.append("\n".join(lines))
        except Exception:
            pass

        # User facts
        try:
            resp = httpx.get(
                f"{HIPP0_BASE_URL}/api/hermes/extracted-facts",
                params={"project_id": project_id, "agent_name": profile.name},
                headers=headers, timeout=10,
            )
            if resp.status_code == 200:
                facts = resp.json().get("facts", [])
                if facts:
                    lines = ["## User Preferences (from HIPP0 memory)", ""]
                    for f in facts:
                        key = f.get("key", "unknown")
                        val = f.get("value", "")
                        if val:
                            lines.append(f"- **{key}**: {val}")
                    sections.append("\n".join(lines))
        except Exception:
            pass

        return sections

    def chat(self, content: str, stream_callback=None) -> str:
        """Run a chat turn (called in worker thread). Returns final response."""
        if self._agent is None:
            raise RuntimeError("Agent session not initialized")
        self._interrupted = False
        return self._agent.chat(content, stream_callback=stream_callback)

    def interrupt(self) -> None:
        """Signal the agent to stop the current generation."""
        self._interrupted = True
        if self._agent is not None:
            try:
                self._agent._interrupted = True
            except Exception:
                pass

    def cleanup(self) -> None:
        """End HIPP0 session and close resources."""
        if self._provider and self._agent_loop:
            try:
                self._agent_loop.run_until_complete(self._provider.end_session())
            except Exception:
                pass
            try:
                self._agent_loop.run_until_complete(self._provider.aclose())
            except Exception:
                pass
            self._agent_loop.close()


# ---------------------------------------------------------------------------
# Per-connection state
# ---------------------------------------------------------------------------

class _ConnectionState:
    """Tracks per-WebSocket-connection state: agent sessions and streaming."""

    def __init__(self, hipp0_key: str, ws_loop: asyncio.AbstractEventLoop):
        self.sessions: Dict[str, _AgentSession] = {}
        self._hipp0_key = hipp0_key
        self._ws_loop = ws_loop
        self._current_task: Optional[asyncio.Task] = None

    def get_or_create_session(self, agent_name: str) -> _AgentSession:
        if agent_name not in self.sessions:
            self.sessions[agent_name] = _AgentSession(
                agent_name, self._hipp0_key, self._ws_loop,
            )
        return self.sessions[agent_name]

    def reset_session(self, agent_name: str) -> None:
        if agent_name in self.sessions:
            self.sessions[agent_name].cleanup()
            del self.sessions[agent_name]

    def cleanup_all(self) -> None:
        for session in self.sessions.values():
            session.cleanup()
        self.sessions.clear()


# ---------------------------------------------------------------------------
# WebSocket handler
# ---------------------------------------------------------------------------

# Thread pool for running blocking AIAgent.chat() calls
_executor = None


def _get_executor():
    global _executor
    if _executor is None:
        from concurrent.futures import ThreadPoolExecutor
        _executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="hermes-agent")
    return _executor


async def _send_json(ws, data: dict) -> None:
    """Send a JSON message, swallowing errors if the connection is gone."""
    try:
        await ws.send(json.dumps(data))
    except (websockets.exceptions.ConnectionClosed, RuntimeError):
        pass


async def _handle_message(
    ws,
    msg: dict,
    state: _ConnectionState,
    loop: asyncio.AbstractEventLoop,
) -> None:
    """Handle a 'message' type from the client."""
    agent_name = msg.get("agent_name", "alice")
    content = msg.get("content", "").strip()
    if not content:
        await _send_json(ws, {"type": "error", "message": "Empty message", "recoverable": True})
        return

    # Validate agent exists
    try:
        get_agent(agent_name)
    except AgentNotFoundError:
        await _send_json(ws, {
            "type": "error",
            "message": f"Agent '{agent_name}' not found. Available: {', '.join(list_agents())}",
            "recoverable": True,
        })
        return

    session = state.get_or_create_session(agent_name)
    conversation_id = session.conversation_id

    # Send stream_start
    await _send_json(ws, {
        "type": "stream_start",
        "conversation_id": conversation_id,
        "agent_name": agent_name,
        "model": session._model,
    })

    start_time = time.time()
    token_buf = {"chunks": 0}

    # Stream callback — sends deltas back over WebSocket from worker thread
    def on_delta(text: str):
        if session._interrupted:
            return
        token_buf["chunks"] += 1
        # Schedule the send on the event loop from the worker thread
        asyncio.run_coroutine_threadsafe(
            _send_json(ws, {"type": "stream_delta", "content": text}),
            loop,
        )

    # Run the blocking chat() in a thread
    error_msg = None
    try:
        def _run_chat():
            # Ensure the session is initialized (first call sets up AIAgent)
            if session._agent is None:
                session.setup()
            return session.chat(content, stream_callback=on_delta)

        final_response = await loop.run_in_executor(_get_executor(), _run_chat)
    except Exception as e:
        error_msg = str(e)
        logger.exception("Agent chat error for %s", agent_name)
        final_response = None

    duration = time.time() - start_time

    if error_msg:
        await _send_json(ws, {
            "type": "error",
            "message": f"Agent error: {error_msg[:500]}",
            "recoverable": True,
        })

    # Send stream_end
    await _send_json(ws, {
        "type": "stream_end",
        "conversation_id": conversation_id,
        "tokens": {"chunks": token_buf["chunks"]},
        "duration_seconds": round(duration, 2),
    })


async def _handle_command(
    ws,
    msg: dict,
    state: _ConnectionState,
) -> None:
    """Handle a 'command' type from the client."""
    command = msg.get("command", "")
    agent_name = msg.get("agent_name", "alice")

    if command == "new":
        state.reset_session(agent_name)
        await _send_json(ws, {
            "type": "stream_start",
            "conversation_id": str(uuid.uuid4()),
            "agent_name": agent_name,
            "model": "claude-sonnet-4-6",
        })
        await _send_json(ws, {
            "type": "stream_end",
            "conversation_id": "",
            "tokens": {"chunks": 0},
            "duration_seconds": 0,
        })
    elif command == "stop":
        session = state.sessions.get(agent_name)
        if session:
            session.interrupt()
        if state._current_task and not state._current_task.done():
            state._current_task.cancel()
    else:
        await _send_json(ws, {
            "type": "error",
            "message": f"Unknown command: {command}",
            "recoverable": True,
        })


async def _ws_handler(ws) -> None:
    """Handle a single WebSocket connection lifecycle."""
    loop = asyncio.get_event_loop()
    hipp0_key = _load_hipp0_key()
    state = _ConnectionState(hipp0_key, loop)

    remote = getattr(ws, "remote_address", ("?", 0))
    logger.info("WebSocket connected: %s", remote)

    try:
        async for raw in ws:
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await _send_json(ws, {
                    "type": "error",
                    "message": "Invalid JSON",
                    "recoverable": True,
                })
                continue

            msg_type = msg.get("type", "")

            if msg_type == "message":
                # Run message handling as a task so we can cancel it
                task = asyncio.create_task(_handle_message(ws, msg, state, loop))
                state._current_task = task
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            elif msg_type == "command":
                await _handle_command(ws, msg, state)
            else:
                await _send_json(ws, {
                    "type": "error",
                    "message": f"Unknown message type: {msg_type}",
                    "recoverable": True,
                })
    except websockets.exceptions.ConnectionClosed:
        pass
    except Exception:
        logger.exception("WebSocket handler error")
    finally:
        state.cleanup_all()
        logger.info("WebSocket disconnected: %s", remote)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_cached_hipp0_key: Optional[str] = None


def _load_hipp0_key() -> str:
    global _cached_hipp0_key
    if _cached_hipp0_key:
        return _cached_hipp0_key
    try:
        _cached_hipp0_key = Path(HIPP0_API_KEY_FILE).read_text().strip()
    except FileNotFoundError:
        logger.error("HIPP0 API key file not found: %s", HIPP0_API_KEY_FILE)
        sys.exit(1)
    return _cached_hipp0_key


# ---------------------------------------------------------------------------
# Server entry point
# ---------------------------------------------------------------------------

async def _serve() -> None:
    logger.info(
        "Hermes Web Platform starting on port %d (HIPP0: %s)",
        WEB_PLATFORM_PORT,
        HIPP0_BASE_URL,
    )

    # Validate prerequisites
    if not ANTHROPIC_API_KEY:
        logger.error("ANTHROPIC_API_KEY not set")
        sys.exit(1)
    _load_hipp0_key()

    async with websockets.serve(
        _ws_handler,
        "0.0.0.0",
        WEB_PLATFORM_PORT,
        origins=None,  # Allow all origins (Caddy handles CORS)
        ping_interval=30,
        ping_timeout=10,
        max_size=2**20,  # 1 MB max message size
    ):
        logger.info("WebSocket server listening on 0.0.0.0:%d", WEB_PLATFORM_PORT)
        await asyncio.Future()  # run forever


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    try:
        asyncio.run(_serve())
    except KeyboardInterrupt:
        logger.info("Shutting down")


if __name__ == "__main__":
    main()
