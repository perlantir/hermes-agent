"""Entry point for the multi-bot Telegram gateway.

Runs as: ``python -m gateway.telegram_multi_launcher``.

For each ``TELEGRAM_BOT_TOKEN_<NAME>`` env var, starts a dedicated
python-telegram-bot Application bound to the persistent agent named
``<name>``. All Applications share one asyncio event loop.

One bot's polling error must not crash the others — each bot is
started independently and supervised; exceptions are logged and the
remaining bots keep running.
"""

from __future__ import annotations

import asyncio
import logging
import signal
import sys
from pathlib import Path
from typing import List

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from gateway.platforms.telegram_multi import (
    SESSION_SWEEP_INTERVAL_SECONDS,
    AgentBot,
    build_bots,
)

logger = logging.getLogger("hermes.telegram_multi_launcher")


async def _run_bot(bot: AgentBot) -> None:
    """Start one bot's Application and keep it polling."""
    app = bot.build()
    try:
        await app.initialize()
        await app.start()
        await app.updater.start_polling(drop_pending_updates=True)
        logger.info("[%s] polling started", bot.agent_name)
        # Park forever — cancellation stops us.
        while True:
            await asyncio.sleep(3600)
    except asyncio.CancelledError:
        raise
    except Exception as e:
        logger.exception("[%s] bot task failed: %s", bot.agent_name, e)
    finally:
        try:
            if app.updater and app.updater.running:
                await app.updater.stop()
        except Exception:
            pass
        try:
            if app.running:
                await app.stop()
        except Exception:
            pass
        try:
            await app.shutdown()
        except Exception:
            pass
        try:
            await bot.shutdown_sessions()
        except Exception:
            pass


async def _idle_sweep(bots: List[AgentBot]) -> None:
    while True:
        try:
            await asyncio.sleep(SESSION_SWEEP_INTERVAL_SECONDS)
        except asyncio.CancelledError:
            return
        for bot in bots:
            try:
                await bot.sweep_idle_sessions()
            except Exception as e:
                logger.warning("[%s] sweep failed: %s", bot.agent_name, e)


async def _main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    loop = asyncio.get_running_loop()
    bots = await build_bots(loop)
    if not bots:
        logger.error("no bots discovered — set TELEGRAM_BOT_TOKEN_<NAME> env vars")
        return 1

    names = ", ".join(b.agent_name for b in bots)
    logger.info("[telegram-multi] Starting %d agents: %s", len(bots), names)

    stop_event = asyncio.Event()

    def _signal() -> None:
        logger.info("shutdown signal received")
        stop_event.set()

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, _signal)
        except NotImplementedError:  # pragma: no cover — windows
            pass

    bot_tasks = [asyncio.create_task(_run_bot(b), name=f"bot:{b.agent_name}") for b in bots]
    sweeper = asyncio.create_task(_idle_sweep(bots), name="idle-sweep")

    await stop_event.wait()

    for t in bot_tasks + [sweeper]:
        t.cancel()
    await asyncio.gather(*bot_tasks, sweeper, return_exceptions=True)
    logger.info("[telegram-multi] shutdown complete")
    return 0


def main() -> None:
    sys.exit(asyncio.run(_main()))


if __name__ == "__main__":
    main()
