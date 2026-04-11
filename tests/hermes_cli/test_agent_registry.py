"""Tests for ``hermes_cli.agent_registry`` — persistent named agents.

These tests rely on ``conftest._isolate_hermes_home`` which redirects
``HERMES_HOME`` to a per-test temp dir. ``get_default_hermes_root()``
resolves to that temp dir because it's outside ``~/.hermes``.
"""

from __future__ import annotations

import io
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import pytest
import yaml

from hermes_cli import agent_registry
from hermes_cli.agent_registry import (
    AgentAlreadyExistsError,
    AgentConfig,
    AgentConfigError,
    AgentNotFoundError,
    AgentProfile,
    InvalidAgentNameError,
    _main,
    agent_exists,
    agents_root,
    delete_agent,
    get_agent,
    get_agent_dir,
    list_agents,
    register_agent,
    update_agent_config,
    validate_agent_name,
)


# ---------------------------------------------------------------------------
# validate_agent_name
# ---------------------------------------------------------------------------


class TestValidateAgentName:
    @pytest.mark.parametrize(
        "name",
        [
            "alice",
            "bob",
            "a",
            "a1",
            "sales-lead",
            "agent_42",
            "x" + "y" * 63,  # 64 chars
        ],
    )
    def test_accepts_valid(self, name):
        validate_agent_name(name)  # should not raise

    @pytest.mark.parametrize(
        "name",
        [
            "",
            "Alice",            # uppercase
            "1alice",           # leading digit
            "-alice",           # leading dash
            "_alice",           # leading underscore
            "alice!",           # bad char
            "alice space",      # whitespace
            "a" * 65,           # too long
            "alice/bob",        # path separator
            None,               # wrong type
            42,                 # wrong type
        ],
    )
    def test_rejects_invalid(self, name):
        with pytest.raises(InvalidAgentNameError):
            validate_agent_name(name)


# ---------------------------------------------------------------------------
# AgentConfig
# ---------------------------------------------------------------------------


class TestAgentConfig:
    def test_defaults(self):
        cfg = AgentConfig()
        assert cfg.model.startswith("anthropic/")
        assert cfg.toolset == "default"
        assert cfg.platform_access == ["cli"]
        assert cfg.project_id is None
        assert cfg.agent_id is None
        assert cfg.extra == {}

    def test_from_dict_roundtrip_preserves_unknown_keys(self):
        raw = {
            "model": "openai/gpt-5",
            "toolset": "coding",
            "platform_access": ["cli", "telegram"],
            "project_id": "proj-uuid",
            "agent_id": "agent-uuid",
            "custom_future_field": 123,
        }
        cfg = AgentConfig.from_dict(raw)
        assert cfg.model == "openai/gpt-5"
        assert cfg.toolset == "coding"
        assert cfg.platform_access == ["cli", "telegram"]
        assert cfg.project_id == "proj-uuid"
        assert cfg.agent_id == "agent-uuid"
        assert cfg.extra == {"custom_future_field": 123}

        out = cfg.to_dict()
        assert out["custom_future_field"] == 123
        assert out["model"] == "openai/gpt-5"

    def test_from_dict_rejects_non_mapping(self):
        with pytest.raises(AgentConfigError):
            AgentConfig.from_dict(["not", "a", "mapping"])  # type: ignore[arg-type]

    def test_from_dict_rejects_bad_platform_access(self):
        with pytest.raises(AgentConfigError):
            AgentConfig.from_dict({"platform_access": "cli"})  # must be list
        with pytest.raises(AgentConfigError):
            AgentConfig.from_dict({"platform_access": [1, 2]})  # must be strs

    def test_from_dict_none(self):
        cfg = AgentConfig.from_dict(None)
        assert cfg.platform_access == ["cli"]


# ---------------------------------------------------------------------------
# register_agent + get_agent + list_agents
# ---------------------------------------------------------------------------


class TestRegisterAndLoad:
    def test_register_writes_soul_and_config(self):
        profile = register_agent(
            "alice",
            soul="# Alice\n\nSales lead persona.\n",
        )
        assert isinstance(profile, AgentProfile)
        assert profile.name == "alice"
        assert profile.root == agents_root() / "alice"
        assert profile.soul_path.read_text(encoding="utf-8").startswith("# Alice")
        assert profile.config_path.is_file()

        raw = yaml.safe_load(profile.config_path.read_text(encoding="utf-8"))
        assert raw["model"].startswith("anthropic/")
        assert raw["platform_access"] == ["cli"]
        assert raw["project_id"] is None
        assert raw["agent_id"] is None

    def test_register_with_custom_config(self):
        cfg = AgentConfig(
            model="openai/gpt-5",
            toolset="coding",
            platform_access=["cli", "telegram"],
            project_id="proj-uuid",
        )
        profile = register_agent("bob", soul="# Bob\n", config=cfg)
        assert profile.config.model == "openai/gpt-5"
        assert profile.config.toolset == "coding"
        assert profile.config.platform_access == ["cli", "telegram"]
        assert profile.config.project_id == "proj-uuid"

    def test_register_rejects_duplicate_without_overwrite(self):
        register_agent("alice", soul="v1\n")
        with pytest.raises(AgentAlreadyExistsError):
            register_agent("alice", soul="v2\n")

    def test_register_overwrite_replaces_soul(self):
        register_agent("alice", soul="v1\n")
        register_agent("alice", soul="v2\n", overwrite=True)
        assert get_agent("alice").soul == "v2\n"

    def test_register_rejects_invalid_name(self):
        with pytest.raises(InvalidAgentNameError):
            register_agent("BadName", soul="x")

    def test_get_agent_raises_when_dir_missing(self):
        with pytest.raises(AgentNotFoundError):
            get_agent("ghost")

    def test_get_agent_raises_when_soul_missing(self):
        dir_ = get_agent_dir("headless")
        dir_.mkdir(parents=True)
        # No SOUL.md
        with pytest.raises(AgentNotFoundError):
            get_agent("headless")

    def test_get_agent_reads_memory_md_when_present(self):
        register_agent("alice", soul="# Alice\n")
        memory_path = agents_root() / "alice" / "MEMORY.md"
        memory_path.write_text("# rolling summary\n", encoding="utf-8")
        profile = get_agent("alice")
        assert profile.memory == "# rolling summary\n"

    def test_get_agent_handles_missing_memory_md(self):
        register_agent("alice", soul="# Alice\n")
        profile = get_agent("alice")
        assert profile.memory == ""

    def test_get_agent_raises_on_bad_yaml(self):
        register_agent("alice", soul="# Alice\n")
        (agents_root() / "alice" / "config.yaml").write_text(
            "this: is: not: valid:::yaml", encoding="utf-8"
        )
        with pytest.raises(AgentConfigError):
            get_agent("alice")

    def test_list_agents_empty(self):
        assert list_agents() == []

    def test_list_agents_sorted_and_filtered(self):
        register_agent("zeta", soul="z")
        register_agent("alpha", soul="a")
        register_agent("mike", soul="m")

        # Dir without a SOUL.md should be ignored
        (agents_root() / "headless").mkdir()
        # Dir whose name doesn't match the regex should be ignored
        (agents_root() / "Bad_Name").mkdir()
        (agents_root() / "Bad_Name" / "SOUL.md").write_text("x")

        assert list_agents() == ["alpha", "mike", "zeta"]

    def test_agent_exists(self):
        assert not agent_exists("alice")
        register_agent("alice", soul="x")
        assert agent_exists("alice")
        assert not agent_exists("BadName")  # invalid name -> False, not raise


# ---------------------------------------------------------------------------
# update_agent_config / delete_agent
# ---------------------------------------------------------------------------


class TestUpdateAndDelete:
    def test_update_agent_config_merges(self):
        register_agent("alice", soul="x")
        updated = update_agent_config(
            "alice", agent_id="new-uuid", project_id="proj-uuid"
        )
        assert updated.config.agent_id == "new-uuid"
        assert updated.config.project_id == "proj-uuid"
        # Unchanged fields stay put
        assert updated.config.toolset == "default"

    def test_update_preserves_extra_keys(self):
        register_agent(
            "alice",
            soul="x",
            config=AgentConfig(
                extra={"telemetry_tag": "hermes-1"},
            ),
        )
        updated = update_agent_config("alice", agent_id="uuid")
        assert updated.config.extra.get("telemetry_tag") == "hermes-1"

    def test_delete_agent(self):
        register_agent("alice", soul="x")
        delete_agent("alice")
        assert not agent_exists("alice")
        with pytest.raises(AgentNotFoundError):
            delete_agent("alice")
        delete_agent("alice", missing_ok=True)  # no raise


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


class TestCLI:
    def test_list_command_prints_agents(self, capsys):
        register_agent("alice", soul="a")
        register_agent("bob", soul="b")

        rc = _main(["list"])
        captured = capsys.readouterr()
        assert rc == 0
        assert captured.out.splitlines() == ["alice", "bob"]

    def test_list_default_command(self, capsys):
        register_agent("alice", soul="a")
        rc = _main([])  # no subcommand -> list
        captured = capsys.readouterr()
        assert rc == 0
        assert "alice" in captured.out

    def test_list_empty(self, capsys):
        rc = _main(["list"])
        captured = capsys.readouterr()
        assert rc == 0
        assert "(no agents)" in captured.out

    def test_show_command_renders_config_and_soul(self, capsys):
        register_agent("alice", soul="# Alice soul\n")
        rc = _main(["show", "alice"])
        captured = capsys.readouterr()
        assert rc == 0
        assert "# Agent: alice" in captured.out
        assert "## config.yaml" in captured.out
        assert "## SOUL.md" in captured.out
        assert "# Alice soul" in captured.out

    def test_show_missing_agent_exits_nonzero(self, capsys):
        rc = _main(["show", "ghost"])
        captured = capsys.readouterr()
        assert rc == 1
        assert "error" in captured.err.lower()


# ---------------------------------------------------------------------------
# Example profiles in the repo are loadable
# ---------------------------------------------------------------------------


class TestExampleProfiles:
    def test_examples_have_valid_configs(self):
        # Locate the repo root via the agent_registry module file.
        repo_root = Path(agent_registry.__file__).resolve().parents[1]
        examples = repo_root / "examples" / "agents"
        assert examples.is_dir(), f"expected examples/agents at {examples}"

        for name in ("alice", "bob"):
            soul = (examples / name / "SOUL.md").read_text(encoding="utf-8")
            cfg = yaml.safe_load(
                (examples / name / "config.yaml").read_text(encoding="utf-8")
            )
            # Both should parse into a valid AgentConfig
            parsed = AgentConfig.from_dict(cfg)
            assert parsed.model
            assert "cli" in parsed.platform_access
            # Name regex should accept the directory name
            validate_agent_name(name)
            assert len(soul) > 10
