# Example Persistent Agents

These are starter profiles for the `feat/persistent-agents-hipp0`
workstream. Copy them into `<hermes_root>/agents/<name>/` (or register
programmatically via `hermes_cli.agent_registry.register_agent`) to try
the persistent-agent / HIPP0 memory integration.

```
examples/agents/
  alice/                # sales lead
    SOUL.md
    config.yaml
  bob/                  # product engineer
    SOUL.md
    config.yaml
```

## Quick start

```bash
# Copy the examples into your local agent registry
mkdir -p ~/.hermes/agents
cp -r examples/agents/alice ~/.hermes/agents/alice
cp -r examples/agents/bob   ~/.hermes/agents/bob

# List what's registered
python -m hermes_cli.agent_registry list
# alice
# bob

# Inspect one
python -m hermes_cli.agent_registry show alice
```

Once HIPP0 is reachable (Phase H2), the `Hipp0MemoryProvider` will call
`POST /api/hermes/register` on first use and write the returned
`agent_id` back to each agent's `config.yaml`. Until then the example
configs leave `project_id` and `agent_id` as `null`.

See [`CLAUDE.md`](../../CLAUDE.md) for agent coding conventions and
[`HIPP0_REQUESTS.md`](../../HIPP0_REQUESTS.md) for any requested
contract changes to the sibling HIPP0 repo.
