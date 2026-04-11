"""Persistent named agent registry.

Part of the ``feat/persistent-agents-hipp0`` workstream: turns ephemeral
Hermes sessions into long-lived named agents (alice, bob, …) whose
identity persists on disk and whose memory is backed by a HIPP0 project.

On-disk layout (locked — see CLAUDE.md / HIPP0 contract notes)::

    <hermes_root>/agents/<name>/
        SOUL.md         # human-edited persona / role / instructions
        MEMORY.md       # READ-ONLY projection from HIPP0, refreshed
                        # at session start by Hipp0MemoryProvider
        config.yaml     # model, toolset, platform_access, project_id,
                        # agent_id (and any future fields)
        pending.jsonl   # WAL for HTTP calls that failed during HIPP0
                        # outages — managed by Hipp0MemoryProvider
        hermes.pid      # daemon PID for graceful restarts (managed
                        # by the gateway runner)

The ``<hermes_root>`` is anchored to ``get_default_hermes_root()`` so
that agents are shared across Hermes *profiles* (a profile is a
HERMES_HOME sandbox; an agent is a persona that any profile can load).
This mirrors how ``hermes_cli.profiles`` anchors the profile directory.

Agent names validate against ``^[a-z][a-z0-9_-]{0,63}$`` (task spec,
subtly stricter than ``profiles.py`` which also allows a leading digit
— agents start with a letter so they parse cleanly as Telegram
``@mentions`` and CLI positional arguments).

This module deliberately makes **no** HTTP calls. Registering an agent
writes SOUL.md + config.yaml locally; the HIPP0 round-trip that
assigns an ``agent_id`` is done separately by
``agent.hipp0_memory_provider.Hipp0MemoryProvider`` in Phase H2, and
the result is persisted back here via :func:`update_agent_config`.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from hermes_constants import get_default_hermes_root
from utils import atomic_yaml_write

# Agent name regex — subtly stricter than the profile regex (which allows
# a leading digit): agents start with a letter so Telegram @mentions and
# CLI positional arguments parse unambiguously.
AGENT_NAME_RE = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")

SOUL_FILENAME = "SOUL.md"
MEMORY_FILENAME = "MEMORY.md"
CONFIG_FILENAME = "config.yaml"
PENDING_WAL_FILENAME = "pending.jsonl"
PID_FILENAME = "hermes.pid"

_DEFAULT_MODEL = "anthropic/claude-opus-4.6"
_DEFAULT_TOOLSET = "default"
_DEFAULT_PLATFORM_ACCESS: tuple[str, ...] = ("cli",)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class AgentRegistryError(Exception):
    """Base class for agent registry errors."""


class AgentNotFoundError(AgentRegistryError):
    """Raised when a named agent does not exist on disk."""


class InvalidAgentNameError(AgentRegistryError, ValueError):
    """Raised when an agent name fails validation."""


class AgentAlreadyExistsError(AgentRegistryError):
    """Raised when registering an agent that already has a SOUL.md."""


class AgentConfigError(AgentRegistryError):
    """Raised when an agent's config.yaml is malformed."""


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class AgentConfig:
    """Typed view over an agent's ``config.yaml``.

    Forward-compatible: unknown keys in the YAML are preserved in
    :attr:`extra` and round-tripped on write so newer Hermes revisions
    can add fields without clobbering older ones.
    """

    model: str = _DEFAULT_MODEL
    toolset: str = _DEFAULT_TOOLSET
    platform_access: List[str] = field(
        default_factory=lambda: list(_DEFAULT_PLATFORM_ACCESS)
    )
    project_id: Optional[str] = None  # HIPP0 tenant UUID
    agent_id: Optional[str] = None  # HIPP0 agent UUID, set after register()
    extra: Dict[str, Any] = field(default_factory=dict)

    _KNOWN_KEYS = frozenset(
        {"model", "toolset", "platform_access", "project_id", "agent_id"}
    )

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "AgentConfig":
        if data is None:
            return cls()
        if not isinstance(data, dict):
            raise AgentConfigError(
                f"Agent config must be a mapping, got {type(data).__name__}"
            )
        platform_access = data.get("platform_access")
        if platform_access is None:
            platform_access = list(_DEFAULT_PLATFORM_ACCESS)
        elif not isinstance(platform_access, list) or not all(
            isinstance(p, str) for p in platform_access
        ):
            raise AgentConfigError(
                "platform_access must be a list of strings"
            )

        extra = {k: v for k, v in data.items() if k not in cls._KNOWN_KEYS}
        return cls(
            model=str(data.get("model", _DEFAULT_MODEL)),
            toolset=str(data.get("toolset", _DEFAULT_TOOLSET)),
            platform_access=list(platform_access),
            project_id=data.get("project_id"),
            agent_id=data.get("agent_id"),
            extra=extra,
        )

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "model": self.model,
            "toolset": self.toolset,
            "platform_access": list(self.platform_access),
            "project_id": self.project_id,
            "agent_id": self.agent_id,
        }
        out.update(self.extra)
        return out


@dataclass
class AgentProfile:
    """Loaded view of an agent: paths, SOUL, MEMORY, config."""

    name: str
    root: Path
    soul: str
    config: AgentConfig
    memory: str = ""  # MEMORY.md contents — empty string if file absent

    @property
    def soul_path(self) -> Path:
        return self.root / SOUL_FILENAME

    @property
    def memory_path(self) -> Path:
        return self.root / MEMORY_FILENAME

    @property
    def config_path(self) -> Path:
        return self.root / CONFIG_FILENAME

    @property
    def pending_wal_path(self) -> Path:
        return self.root / PENDING_WAL_FILENAME


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def agents_root() -> Path:
    """Return the directory where named agents are stored.

    Anchored to the hermes root (not HERMES_HOME) so agents are shared
    across profiles. Mirrors :func:`hermes_cli.profiles._get_profiles_root`.
    """
    return get_default_hermes_root() / "agents"


def get_agent_dir(name: str) -> Path:
    """Return the on-disk directory for an agent (without validating existence)."""
    validate_agent_name(name)
    return agents_root() / name


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_agent_name(name: str) -> None:
    """Raise :class:`InvalidAgentNameError` if *name* is not a valid identifier."""
    if not isinstance(name, str) or not AGENT_NAME_RE.match(name):
        raise InvalidAgentNameError(
            f"Invalid agent name {name!r}. Must match ^[a-z][a-z0-9_-]{{0,63}}$"
        )


def agent_exists(name: str) -> bool:
    """Return True iff an agent with a SOUL.md exists on disk."""
    try:
        validate_agent_name(name)
    except InvalidAgentNameError:
        return False
    dir_ = agents_root() / name
    return dir_.is_dir() and (dir_ / SOUL_FILENAME).is_file()


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------


def list_agents() -> List[str]:
    """List registered agent names, sorted. Entries without SOUL.md are skipped."""
    root = agents_root()
    if not root.is_dir():
        return []
    out: List[str] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        try:
            validate_agent_name(child.name)
        except InvalidAgentNameError:
            # Ignore stray dirs whose name doesn't match the agent regex.
            continue
        if (child / SOUL_FILENAME).is_file():
            out.append(child.name)
    return out


def get_agent(name: str) -> AgentProfile:
    """Load a registered agent from disk.

    Raises :class:`AgentNotFoundError` if the directory or SOUL.md is
    missing, :class:`AgentConfigError` if config.yaml is malformed.
    """
    validate_agent_name(name)
    dir_ = agents_root() / name
    if not dir_.is_dir():
        raise AgentNotFoundError(f"Agent {name!r} does not exist at {dir_}")

    soul_path = dir_ / SOUL_FILENAME
    if not soul_path.is_file():
        raise AgentNotFoundError(
            f"Agent {name!r} is missing SOUL.md at {soul_path}"
        )
    soul = soul_path.read_text(encoding="utf-8")

    config_path = dir_ / CONFIG_FILENAME
    if config_path.is_file():
        try:
            raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        except yaml.YAMLError as e:
            raise AgentConfigError(
                f"Agent {name!r} config.yaml is not valid YAML: {e}"
            ) from e
        config = AgentConfig.from_dict(raw or {})
    else:
        config = AgentConfig()

    memory = ""
    memory_path = dir_ / MEMORY_FILENAME
    if memory_path.is_file():
        memory = memory_path.read_text(encoding="utf-8")

    return AgentProfile(
        name=name,
        root=dir_,
        soul=soul,
        config=config,
        memory=memory,
    )


# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------


def register_agent(
    name: str,
    soul: str,
    config: Optional[AgentConfig] = None,
    *,
    overwrite: bool = False,
) -> AgentProfile:
    """Create or update an agent on disk.

    Does **not** call HIPP0. Phase H2's :class:`Hipp0MemoryProvider`
    posts to ``/api/hermes/register`` and then calls
    :func:`update_agent_config` to persist the returned ``agent_id``.

    Args:
        name: agent identifier matching ``^[a-z][a-z0-9_-]{0,63}$``.
        soul: full text of the agent's SOUL.md (persona/instructions).
        config: optional typed :class:`AgentConfig`. Defaults applied if None.
        overwrite: if False, refuses to overwrite an agent that already has
            a SOUL.md on disk.

    Returns:
        A freshly loaded :class:`AgentProfile` reflecting the on-disk state.
    """
    validate_agent_name(name)
    if not isinstance(soul, str):
        raise TypeError("soul must be a string")

    dir_ = agents_root() / name
    soul_exists = (dir_ / SOUL_FILENAME).is_file()
    if soul_exists and not overwrite:
        raise AgentAlreadyExistsError(
            f"Agent {name!r} already exists at {dir_}. "
            "Pass overwrite=True to replace."
        )

    dir_.mkdir(parents=True, exist_ok=True)

    # SOUL.md is plain text — write atomically via temp-file + replace to
    # match atomic_yaml_write's crash-safety guarantees.
    _atomic_text_write(dir_ / SOUL_FILENAME, soul)

    cfg = config or AgentConfig()
    atomic_yaml_write(dir_ / CONFIG_FILENAME, cfg.to_dict())

    return get_agent(name)


def update_agent_config(name: str, **updates: Any) -> AgentProfile:
    """Merge-update an agent's config.yaml.

    Unknown keys are preserved in :attr:`AgentConfig.extra`. Pass
    ``platform_access=[...]`` to replace the list (no element-level merge).
    """
    profile = get_agent(name)
    merged = profile.config.to_dict()
    merged.update(updates)
    new_cfg = AgentConfig.from_dict(merged)
    atomic_yaml_write(profile.config_path, new_cfg.to_dict())
    return get_agent(name)


def delete_agent(name: str, *, missing_ok: bool = False) -> None:
    """Remove an agent's directory from disk.

    ``missing_ok=True`` suppresses :class:`AgentNotFoundError` when the
    directory is already gone.
    """
    validate_agent_name(name)
    dir_ = agents_root() / name
    if not dir_.exists():
        if missing_ok:
            return
        raise AgentNotFoundError(f"Agent {name!r} does not exist at {dir_}")

    import shutil

    shutil.rmtree(dir_)


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _atomic_text_write(path: Path, content: str) -> None:
    """Atomically write text via temp-file + os.replace.

    Mirrors ``utils.atomic_yaml_write`` for non-YAML payloads (SOUL.md).
    Keeping this local avoids adding a sixth file to this phase while
    still honoring the atomic-write convention the rest of Hermes uses.
    """
    import os
    import tempfile

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp_path = tempfile.mkstemp(
        dir=str(path.parent),
        prefix=f".{path.stem}_",
        suffix=".tmp",
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


# ---------------------------------------------------------------------------
# CLI entry point: ``python -m hermes_cli.agent_registry [list|show NAME]``
# ---------------------------------------------------------------------------


def _main(argv: Optional[List[str]] = None) -> int:
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        prog="python -m hermes_cli.agent_registry",
        description="Inspect the local persistent agent registry.",
    )
    sub = parser.add_subparsers(dest="command")
    sub.add_parser(
        "list", help="List agents registered in <hermes_root>/agents/"
    )
    show = sub.add_parser("show", help="Print an agent's config + SOUL")
    show.add_argument("name")

    args = parser.parse_args(argv)

    if args.command in (None, "list"):
        names = list_agents()
        if not names:
            print("(no agents)")
            return 0
        for n in names:
            print(n)
        return 0

    if args.command == "show":
        try:
            agent = get_agent(args.name)
        except AgentRegistryError as e:
            print(f"error: {e}", file=sys.stderr)
            return 1
        print(f"# Agent: {agent.name}")
        print(f"# Root:  {agent.root}")
        print()
        print("## config.yaml")
        print(yaml.safe_dump(agent.config.to_dict(), sort_keys=False).rstrip())
        print()
        print("## SOUL.md")
        print(agent.soul.rstrip())
        return 0

    parser.print_help()
    return 2


if __name__ == "__main__":  # pragma: no cover - CLI trampoline
    raise SystemExit(_main())
