"""Multi-bot Telegram gateway — one bot per persistent agent.

Each TELEGRAM_BOT_TOKEN_<NAME> env var is mapped to a single persistent
agent at ``~/.hermes/agents/<name>/``. A single python-telegram-bot
Application is created per token, all run concurrently on the same
asyncio event loop.

Memory integration follows the same pattern as
``hermes_cli/repl.py``:

  - HIPP0 session is started per ``(chat_id, agent_name)`` pair.
  - ``Hipp0MemoryProvider`` is wrapped in a sync adapter and injected
    into ``AIAgent._memory_manager`` so ``compile`` runs before each
    turn and ``capture`` runs after.
  - ``agent.chat()`` runs in a worker thread; the sync adapter bridges
    back to the main asyncio loop via ``run_coroutine_threadsafe``.

One process hosts all N bots. An error in one bot's polling/handler
must not crash the others — each Application is started independently
and exceptions are logged but not propagated.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Make the hermes-agent repo importable when run via
# ``python -m gateway.telegram_multi_launcher``.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from telegram import Update
from telegram.constants import ChatAction
from telegram.error import TelegramError
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

try:
    from telegram.ext import MessageReactionHandler
    _MESSAGE_REACTION_HANDLER_AVAILABLE = True
except ImportError:  # older python-telegram-bot
    MessageReactionHandler = None  # type: ignore[assignment]
    _MESSAGE_REACTION_HANDLER_AVAILABLE = False

from hermes_cli.agent_registry import (
    AgentNotFoundError,
    AgentProfile,
    agent_exists,
    get_agent,
)

logger = logging.getLogger("hermes.telegram_multi")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

AGENT_EMOJI: Dict[str, str] = {
    "hipp0": "💬",
    "maks": "⚡",
    "forge": "🔨",
    "chain": "⛓️",
    "scout": "🔍",
    "counsel": "⚖️",
    "launch": "🚀",
    "pixel": "🎨",
    "polish": "✨",
    "sentinel": "🛡️",
    "relay": "🔄",
    "aegis": "🛡",
}

SESSION_IDLE_TIMEOUT_SECONDS = 2 * 60 * 60  # 2 hours
SESSION_SWEEP_INTERVAL_SECONDS = 10 * 60    # 10 minutes
TELEGRAM_MAX_MESSAGE = 4096
TELEGRAM_SAFE_CHUNK = 3900  # leave headroom for prefix/formatting

TOKEN_ENV_PREFIX = "TELEGRAM_BOT_TOKEN_"


# ---------------------------------------------------------------------------
# Sync adapter — identical pattern to hermes_cli/repl.py
# ---------------------------------------------------------------------------


class _Hipp0SyncAdapter:
    """Bridge the async Hipp0MemoryProvider into MemoryManager's sync ABC.

    ``agent.chat()`` runs in a worker thread (``asyncio.to_thread``).
    The sync adapter is called from that worker thread and schedules
    provider coroutines onto the main asyncio loop via
    ``asyncio.run_coroutine_threadsafe``, blocking the worker until
    the coroutine completes.
    """

    def __init__(self, provider: Any, main_loop: asyncio.AbstractEventLoop) -> None:
        self._provider = provider
        self._loop = main_loop

    @property
    def name(self) -> str:
        return "hipp0"

    def is_available(self) -> bool:
        return self._provider.is_available()

    def initialize(self, session_id: str, **kwargs: Any) -> None:
        self._provider.initialize(session_id, **kwargs)

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return []

    def _run(self, coro):
        fut = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return fut.result()

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        if not query:
            return ""
        try:
            compiled = self._run(self._provider.compile(query, fast_mode=True))
            return compiled.as_prompt_block()
        except Exception as e:  # pragma: no cover — degraded mode
            logger.warning("HIPP0 prefetch (compile) failed: %s", e)
            return ""

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        pass

    def sync_turn(
        self, user_content: str, assistant_content: str, *, session_id: str = ""
    ) -> None:
        transcript = f"USER: {user_content}\nASSISTANT: {assistant_content}"
        try:
            self._run(self._provider.capture(transcript, source="hermes"))
        except Exception as e:  # pragma: no cover
            logger.warning("HIPP0 sync_turn (capture) failed: %s", e)

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

    def on_delegation(
        self, task: str, result: str, *, child_session_id: str = "", **kwargs: Any
    ) -> None:
        pass

    def get_config_schema(self) -> List[Dict[str, Any]]:
        return []

    def save_config(self, values: Dict[str, Any], hermes_home: str) -> None:
        pass


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------


@dataclass
class _Session:
    """One (chat_id, agent) conversation: AIAgent + HIPP0 provider."""

    chat_id: int
    agent_name: str
    agent: Any
    provider: Any
    session_id: str
    decision_count: int = 0
    last_activity: float = field(default_factory=time.time)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    # Local SQLite session_id of the most recent agent turn. Outcome
    # signals are recorded against this id (the AIAgent's session_id),
    # not the HIPP0 ``session_id`` above — those live in different tables.
    last_local_session_id: Optional[str] = None
    # Multi-turn conversation history. ``AIAgent.chat()`` is amnesic — it
    # calls ``run_conversation`` with no prior history — so we keep the
    # ``messages`` list returned from each turn and feed it back in.
    history: List[Dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Per-agent bot
# ---------------------------------------------------------------------------


def _split_for_telegram(text: str, chunk_size: int = TELEGRAM_SAFE_CHUNK) -> List[str]:
    """Split *text* into Telegram-sendable chunks at paragraph/line boundaries."""
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]
    chunks: List[str] = []
    remaining = text
    while len(remaining) > chunk_size:
        slice_ = remaining[:chunk_size]
        # Prefer paragraph boundary, then line, then space.
        cut = slice_.rfind("\n\n")
        if cut < chunk_size // 2:
            cut = slice_.rfind("\n")
        if cut < chunk_size // 2:
            cut = slice_.rfind(" ")
        if cut <= 0:
            cut = chunk_size
        chunks.append(remaining[:cut].rstrip())
        remaining = remaining[cut:].lstrip()
    if remaining:
        chunks.append(remaining)
    return chunks


class AgentBot:
    """A single Telegram Application bound to one persistent agent."""

    def __init__(
        self,
        *,
        agent_name: str,
        token: str,
        profile: AgentProfile,
        anthropic_key: str,
        hipp0_base_url: str,
        hipp0_key: str,
        main_loop: asyncio.AbstractEventLoop,
    ) -> None:
        self.agent_name = agent_name
        self.token = token
        self.profile = profile
        self.anthropic_key = anthropic_key
        self.hipp0_base_url = hipp0_base_url
        self.hipp0_key = hipp0_key
        self.main_loop = main_loop
        self.emoji = AGENT_EMOJI.get(agent_name, "🤖")
        self.sessions: Dict[int, _Session] = {}
        self._sessions_lock = asyncio.Lock()
        self.app: Optional[Application] = None

    # --------------------------- lifecycle ---------------------------

    def build(self) -> Application:
        app = (
            ApplicationBuilder()
            .token(self.token)
            .concurrent_updates(True)
            .build()
        )
        app.add_handler(CommandHandler("start", self._cmd_start))
        app.add_handler(CommandHandler("status", self._cmd_status))
        app.add_handler(
            MessageHandler(
                filters.PHOTO | filters.Document.ALL | filters.AUDIO | filters.VOICE | filters.VIDEO,
                self._handle_media,
            )
        )
        app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self._handle_text))
        if _MESSAGE_REACTION_HANDLER_AVAILABLE:
            try:
                app.add_handler(MessageReactionHandler(self._handle_message_reaction))
            except Exception as e:  # pragma: no cover
                logger.warning(
                    "[%s] failed to register MessageReactionHandler: %s",
                    self.agent_name, e,
                )
        app.add_error_handler(self._on_error)
        self.app = app
        return app

    async def _on_error(
        self, update: object, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        logger.exception(
            "[%s] handler error: %s", self.agent_name, context.error
        )

    # --------------------------- commands ----------------------------

    async def _cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        desc = self.profile.config.description or "Persistent Hermes agent."
        await update.effective_chat.send_message(
            f"{self.emoji} {self.agent_name} here. {desc} "
            "Connected to HIPP0 shared memory. Send me a message."
        )

    async def _cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        chat_id = update.effective_chat.id
        sess = self.sessions.get(chat_id)
        hipp0_status = "unknown"
        if sess is not None:
            hipp0_status = "connected" if sess.provider.is_available() else "degraded"
        else:
            # Light-weight probe
            import httpx
            try:
                r = httpx.get(f"{self.hipp0_base_url}/api/health", timeout=3)
                hipp0_status = "reachable" if r.status_code == 200 else f"error {r.status_code}"
            except Exception:
                hipp0_status = "unreachable"
        lines = [
            f"{self.emoji} *{self.agent_name}*",
            f"role: {self.profile.config.role or 'assistant'}",
            f"model: {self.profile.config.model or 'default'}",
            f"HIPP0: {hipp0_status}",
            f"decisions this session: {sess.decision_count if sess else 0}",
        ]
        await update.effective_chat.send_message("\n".join(lines))

    # --------------------------- media -------------------------------

    async def _handle_media(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        await update.effective_chat.send_message(
            f"{self.emoji} I can't process images or files yet — send text."
        )

    # --------------------------- text --------------------------------

    async def _handle_text(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        if not update.message or not update.message.text:
            return
        chat = update.effective_chat
        chat_id = chat.id
        user_text = update.message.text

        if not self.anthropic_key:
            await chat.send_message(
                f"{self.emoji} ANTHROPIC_API_KEY is not configured on the server."
            )
            return

        # Typing indicator.
        try:
            await chat.send_chat_action(ChatAction.TYPING)
        except TelegramError:
            pass

        try:
            session = await self._get_or_create_session(chat_id)
        except Exception as e:
            logger.exception("[%s] session create failed: %s", self.agent_name, e)
            await chat.send_message(
                f"{self.emoji} Failed to start session: {e}"
            )
            return

        # Implicit outcome signal: if this message reads as feedback on the
        # previous exchange, stamp the prior session row before we start a
        # new turn. Fire-and-forget — never block the message flow.
        try:
            self._maybe_record_implicit_outcome(session, user_text)
        except Exception as exc:  # pragma: no cover
            logger.debug("[%s] implicit outcome dispatch: %s", self.agent_name, exc)

        async with session.lock:
            session.last_activity = time.time()
            try:
                # Use run_conversation (not chat()) so we can carry the
                # message history forward across turns. Without this the
                # bot replies as if every message were the first.
                history_in = list(session.history)
                result = await asyncio.to_thread(
                    session.agent.run_conversation,
                    user_text,
                    None,            # system_message — keep agent's ephemeral one
                    history_in,      # conversation_history
                )
            except Exception as e:
                logger.exception("[%s] run_conversation failed: %s", self.agent_name, e)
                await chat.send_message(f"{self.emoji} error: {e}")
                return
            response = (result or {}).get("final_response") or ""
            new_messages = (result or {}).get("messages")
            if isinstance(new_messages, list):
                session.history = new_messages
            session.decision_count += 1
            # Stash the AIAgent's current local session_id so the NEXT
            # incoming message can attach an outcome signal to it.
            try:
                session.last_local_session_id = getattr(
                    session.agent, "session_id", None
                )
            except Exception:
                pass

        if not response:
            response = "(no response)"

        chunks = _split_for_telegram(response)
        first = True
        for chunk in chunks:
            prefix = f"{self.emoji} " if first else ""
            first = False
            try:
                await chat.send_message(prefix + chunk)
            except TelegramError as e:
                logger.warning("[%s] send failed: %s", self.agent_name, e)
                break

    # ----------------------- outcome signals -------------------------

    # Telegram emoji → outcome polarity. Mirrors the map in
    # gateway/platforms/telegram.py so reactions behave consistently
    # across the single-bot and multi-bot adapters.
    _REACTION_OUTCOME_MAP: Dict[str, str] = {
        "\U0001f44d": "positive",   # 👍
        "\u2764":     "positive",   # ❤
        "\u2764\ufe0f": "positive", # ❤️ (with VS16)
        "\U0001f525": "positive",   # 🔥
        "\U0001f31f": "positive",   # 🌟
        "\u2b50":     "positive",   # ⭐
        "\U0001f44e": "negative",   # 👎
        "\U0001f4a9": "negative",   # 💩
        "\U0001f615": "negative",   # 😕
        "\U0001f914": "negative",   # 🤔
    }

    def _maybe_record_implicit_outcome(
        self, session: _Session, text: str
    ) -> None:
        if not text or not session.last_local_session_id:
            return
        from gateway.builtin_hooks.outcome_signals import detect_implicit_outcome
        detected = detect_implicit_outcome([text], [])
        if not detected:
            return
        outcome, source = detected
        target_session_id = session.last_local_session_id
        # Only signal once per agent turn — clear immediately so a follow-up
        # "thanks again" doesn't double-stamp the same row.
        session.last_local_session_id = None
        note = text[:200]
        asyncio.create_task(
            self._record_outcome_async(target_session_id, outcome, source, note)
        )

    async def _record_outcome_async(
        self,
        session_id: str,
        outcome: str,
        signal_source: str,
        note: Optional[str],
    ) -> None:
        try:
            from hermes_state import SessionDB
            db = SessionDB()
            detail = {"signal_source": signal_source, "note": note}
            await asyncio.to_thread(
                db.record_outcome, session_id, outcome, signal_source, detail
            )
            logger.info(
                "[%s] outcome=%s recorded on session %s (source=%s)",
                self.agent_name, outcome, session_id, signal_source,
            )
        except Exception as exc:  # pragma: no cover
            logger.debug(
                "[%s] record_outcome failed for %s: %s",
                self.agent_name, session_id, exc,
            )

    async def _handle_message_reaction(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Fire-and-forget outcome signal from a Telegram reaction.

        Note: Telegram's Bot API only delivers reaction updates in groups
        and channels; reactions in 1:1 DMs are not delivered to bots.
        """
        try:
            reaction_update = getattr(update, "message_reaction", None)
            if reaction_update is None:
                return
            new_reactions = getattr(reaction_update, "new_reaction", None) or []
            chat = getattr(reaction_update, "chat", None)
            chat_id = chat.id if chat and chat.id is not None else None
            if chat_id is None or not new_reactions:
                return
            outcome: Optional[str] = None
            for r in new_reactions:
                emoji = getattr(r, "emoji", None)
                if emoji and emoji in self._REACTION_OUTCOME_MAP:
                    outcome = self._REACTION_OUTCOME_MAP[emoji]
                    break
            if not outcome:
                return
            session = self.sessions.get(chat_id)
            if session is None or not session.last_local_session_id:
                return
            target = session.last_local_session_id
            session.last_local_session_id = None
            asyncio.create_task(
                self._record_outcome_async(
                    target, outcome, "user_reaction", "telegram_reaction"
                )
            )
        except Exception as exc:  # pragma: no cover
            logger.debug("[%s] reaction handler error: %s", self.agent_name, exc)

    # ------------------------- sessions ------------------------------

    async def _get_or_create_session(self, chat_id: int) -> _Session:
        async with self._sessions_lock:
            sess = self.sessions.get(chat_id)
            if sess is not None:
                return sess
            sess = await self._create_session(chat_id)
            self.sessions[chat_id] = sess
            return sess

    async def _create_session(self, chat_id: int) -> _Session:
        from agent.hipp0_memory_provider import Hipp0MemoryProvider, CompiledContext
        from agent.memory_manager import MemoryManager
        from agent.prompt_builder import build_slim_system_prompt
        from run_agent import AIAgent
        from hermes_state import SessionDB

        profile = self.profile
        hermes_home = Path.home() / ".hermes"
        (hermes_home / "memories").mkdir(parents=True, exist_ok=True)

        provider = Hipp0MemoryProvider(
            base_url=self.hipp0_base_url,
            api_key=self.hipp0_key,
            project_id=str(profile.config.project_id),
            agent_name=profile.name,
            agent_id=str(profile.config.agent_id or ""),
            pending_wal_path=profile.pending_wal_path,
            memory_md_path=profile.memory_path,
        )

        try:
            session_id = await provider.start_session(platform="telegram")
        except Exception as e:
            logger.warning(
                "[%s] start_session failed (degraded): %s", self.agent_name, e
            )
            session_id = ""

        try:
            compiled = await provider.compile("General conversation", fast_mode=False)
        except Exception:
            compiled = CompiledContext(
                degraded=True, degraded_reason="compile failed at session start"
            )

        system_prompt = build_slim_system_prompt(
            profile.soul,
            compiled_context_block=compiled.as_prompt_block(),
            platform_hint="telegram",
        )

        session_db = SessionDB()
        agent = AIAgent(
            model=profile.config.model,
            provider="anthropic",
            api_key=self.anthropic_key,
            max_iterations=int(profile.config.extra.get("max_iterations", 50)),
            quiet_mode=True,
            ephemeral_system_prompt=system_prompt,
            platform="telegram",
            skip_context_files=True,
            skip_memory=False,
            slim_prompt=True,
            session_db=session_db,
            agent_name=self.agent_name,
        )

        adapter = _Hipp0SyncAdapter(provider, self.main_loop)
        if agent._memory_manager is None:
            agent._memory_manager = MemoryManager()
        agent._memory_manager.add_provider(adapter)
        adapter.initialize(
            session_id, platform="telegram", hermes_home=str(hermes_home)
        )

        logger.info(
            "[%s] session created chat_id=%s session_id=%s",
            self.agent_name,
            chat_id,
            session_id or "(degraded)",
        )
        return _Session(
            chat_id=chat_id,
            agent_name=self.agent_name,
            agent=agent,
            provider=provider,
            session_id=session_id,
        )

    async def sweep_idle_sessions(self) -> None:
        now = time.time()
        async with self._sessions_lock:
            stale = [
                cid
                for cid, s in self.sessions.items()
                if (now - s.last_activity) > SESSION_IDLE_TIMEOUT_SECONDS
            ]
            for cid in stale:
                sess = self.sessions.pop(cid, None)
                if sess is None:
                    continue
                logger.info(
                    "[%s] idle sweep: closing chat_id=%s", self.agent_name, cid
                )
                try:
                    await sess.provider.end_session()
                except Exception:
                    pass
                try:
                    await sess.provider.aclose()
                except Exception:
                    pass

    async def shutdown_sessions(self) -> None:
        async with self._sessions_lock:
            for sess in list(self.sessions.values()):
                try:
                    await sess.provider.end_session()
                except Exception:
                    pass
                try:
                    await sess.provider.aclose()
                except Exception:
                    pass
            self.sessions.clear()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def discover_bot_configs() -> List[Tuple[str, str]]:
    """Scan env for TELEGRAM_BOT_TOKEN_* vars → list of (agent_name, token).

    Agent names are lowercased and validated against the on-disk agent
    registry; bots whose agent profile is missing are skipped with a log.
    """
    out: List[Tuple[str, str]] = []
    for key, val in os.environ.items():
        if not key.startswith(TOKEN_ENV_PREFIX):
            continue
        token = (val or "").strip()
        if not token:
            continue
        agent_name = key[len(TOKEN_ENV_PREFIX) :].lower()
        if not agent_name:
            continue
        if not agent_exists(agent_name):
            logger.warning(
                "skipping %s: no agent profile at ~/.hermes/agents/%s/",
                key,
                agent_name,
            )
            continue
        out.append((agent_name, token))
    out.sort(key=lambda p: p[0])
    return out


async def build_bots(main_loop: asyncio.AbstractEventLoop) -> List[AgentBot]:
    """Construct all AgentBot instances from env and disk state."""
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not anthropic_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set")

    hipp0_base_url = os.environ.get("HIPP0_BASE_URL", "http://127.0.0.1:3100")
    hipp0_key_path = os.environ.get(
        "HIPP0_API_KEY_FILE", "/etc/team-hippo/api-key.txt"
    )
    hipp0_key = Path(hipp0_key_path).read_text().strip()

    configs = discover_bot_configs()
    bots: List[AgentBot] = []
    for name, token in configs:
        try:
            profile = get_agent(name)
        except AgentNotFoundError as e:
            logger.warning("skipping %s: %s", name, e)
            continue
        bots.append(
            AgentBot(
                agent_name=name,
                token=token,
                profile=profile,
                anthropic_key=anthropic_key,
                hipp0_base_url=hipp0_base_url,
                hipp0_key=hipp0_key,
                main_loop=main_loop,
            )
        )
    return bots
