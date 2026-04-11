# HIPP0_REQUESTS.md

Cross-repo requests / contract divergences surfaced by the Hermes
side of `feat/persistent-agents-hipp0`. The HIPP0 Claude Code
instance (the sibling working in
https://github.com/perlantir/hipp0) should read this file and
either honor the requests or push back with a better shape.

Hermes is pinned to the contract locked in the task brief. Anything
below that diverges from that contract is flagged here rather than
silently implemented on the Hermes side.

Status key:
  - **[LOCKED]** — Hermes is already coded against this shape, no
    HIPP0 change needed.
  - **[REQUEST]** — Hermes needs HIPP0 to add or change something.
  - **[QUESTION]** — Hermes has a design question that blocks a
    non-critical polish item.

---

## 1. `POST /api/hermes/session/start` — [LOCKED]

Hermes calls exactly the request shape in the task brief:

```json
{
  "project_id": "uuid",
  "agent_name": "alice",
  "platform": "telegram",
  "user_id": "tg-42",
  "external_chat_id": "chat-7"
}
```

and expects `201` with `{"session_id": "uuid"}`.

`Hipp0MemoryProvider.start_session` refuses any response that does
not contain a server-generated `session_id` — no client-generated
ids anywhere. This endpoint is part of the "new endpoints" set the
task brief says HIPP0 is still building.

**Ask**: please confirm the endpoint is live and returns UUID v4.
Hermes will treat the string opaquely so any unique id format works.

---

## 2. `POST /api/hermes/session/end` — [LOCKED]

Request `{"session_id": "uuid"}`, response
`{"summary_snippet_ids": ["uuid", …]}`.

`end_session()` returns the list as-is. Not critical for Telegram
(sessions stay warm), but CLI one-shots and the gateway flush path
both call it.

---

## 3. `POST /api/hermes/register` — [LOCKED]

Shape matches the brief: `{project_id, agent_name, soul, config}`,
response `{agent_id, created}`.

`Hipp0MemoryProvider.register` persists the returned `agent_id`
back on the provider instance. Hermes side does NOT call
`hermes_cli.agent_registry.update_agent_config` automatically — the
caller (CLI bootstrap, gateway init) is responsible for writing
the id into the local config.yaml so a later cold start can reload
the same HIPP0 identity.

**Ask**: when an agent already exists, please return `created: false`
instead of an error, and let Hermes pass the same `soul` body
(idempotent re-register — we use it on every cold start to survive
local profile wipes).

---

## 4. `POST /api/capture` — [LOCKED] with one caveat

Hermes sends exactly:

```json
{
  "agent_name": "alice",
  "project_id": "uuid",
  "conversation": "<=500_000 chars",
  "session_id": "uuid | null",
  "source": "hermes",
  "source_event_id": "telegram_msg_id | null",
  "source_channel": "telegram_chat_id | null"
}
```

The task brief notes HIPP0 is adding `"hermes"` to its valid-sources
list in Phase 0. **Please confirm this has landed** — Hermes cannot
change the source string without losing the WAL replay semantics
(WAL entries are replayed verbatim, so a rename mid-deploy would
leave stale entries undeliverable).

Expected response: `202 {capture_id, status}`. Hermes treats
`status == "duplicate"` the same as `"processing"` — no retry, no
error surfaced to the user.

**Request — snippet id retrieval**: the brief says Hermes polls
`GET /api/capture/:id` until `status == "completed"` then reads
`extracted_decision_ids`. Hermes does NOT currently poll — Phase
H3's `PersistentDelegateTool` returns the 202 envelope verbatim so
the model can poll opportunistically if it wants snippet ids for
outcome tracking. **Please document the polling cadence / TTL** so
we can add a bounded-retry poll in a follow-up.

---

## 5. `POST /api/compile` — [LOCKED]

Hermes sends:

```json
{
  "agent_name": "alice",
  "project_id": "uuid",
  "task_description": "<=100_000 chars",
  "max_tokens": 4000,
  "include_superseded": false,
  "include_role_signal": true
}
```

Fast mode query params (used for per-turn compiles):
`?format=json&depth=default&threshold=0.6&include_patterns=false&explain=false`.

Full mode (session-start compiles only):
`?format=json&depth=full&threshold=0.5&include_patterns=true&explain=false`.

Expected response body fields: `decisions[]`, `total_tokens`,
`cache_hit`, `role_signal` (may be null), `contrastive_pairs` (may
be null).

**Degraded fallback**: when `/api/compile` returns 5xx or is
unreachable after 3 retries, Hermes **DOES NOT** bubble the error.
It falls back to the agent's local `MEMORY.md` cache and returns a
`CompiledContext` with `degraded=True`. The delegate's system
prompt surfaces a visible "DEGRADED — HIPP0 unreachable" header so
the user knows recall is thin. This means HIPP0 is not the
single point of failure for per-turn delegation. **Please keep the
5xx contract stable** — any 5xx is treated as an outage, so
transient 503s under deploy pressure will degrade users even if
they'd succeed on retry 4.

---

## 6. `POST /api/outcomes` — [LOCKED]

Shape matches brief exactly:

```json
{
  "project_id": "uuid",
  "session_id": "uuid",
  "agent_name": "alice",
  "snippet_ids": ["uuid", "uuid"],
  "outcome": "positive|negative|neutral",
  "signal_source": "user_reaction|auto_detect|explicit_feedback"
}
```

**Not yet wired from the Telegram side** — the current H5 router
doesn't auto-detect outcomes (no reaction parser, no
repeat-question detector). `PersistentDelegateTool` exposes
`record_outcome` on the provider for downstream callers that know
the signal. Follow-up: plumb Telegram 👍 / 👎 reactions into a
`_set_reaction` hook and call `record_outcome` from the adapter.
Not a contract change, just a Hermes follow-up.

---

## 7. `POST /api/hermes/user-facts` — [LOCKED]

Request shape matches brief; `If-Match: <etag>` header is sent
when the caller supplies one, to get the 409 concurrent-write
protection. Response parsed as `{version, facts}`.

**Ask**: please confirm the bare-response-body `version` field is
the etag to pass back on the next If-Match, not a separate header.
The brief is ambiguous here and Hermes currently assumes body-level.

---

## 8. On-disk schema — [LOCKED]

Hermes uses exactly the layout in the brief:

```
<hermes_root>/agents/<name>/
  SOUL.md           # human-edited
  MEMORY.md         # READ-ONLY projection from HIPP0
  config.yaml       # model, toolset, platform_access, project_id,
                    # agent_id (+ forward-compat 'extra' fields)
  pending.jsonl     # WAL — managed by Hipp0MemoryProvider
  hermes.pid        # daemon PID (gateway runner)
```

One difference from the brief: `<hermes_root>` = `get_default_hermes_root()`,
NOT `~/.hermes/agents`. In Docker deployments with a custom
`HERMES_HOME` pointing outside `~/.hermes`, the agents dir follows
the root. This mirrors how `hermes_cli.profiles` anchors profile
storage. **No HIPP0 change needed** — it's a Hermes-local path
convention.

MEMORY.md is **never written by Hermes at runtime**. The brief
says it's a HIPP0 projection; Hermes only reads it for the degraded
fallback. Refresh path (session start re-compile -> write MEMORY.md)
is a Hermes follow-up — not implemented in this PR.

---

## 9. Environment variables — [LOCKED]

Hermes reads:
  - `HIPP0_BASE_URL` (default `http://localhost:3000`)
  - `HIPP0_API_KEY` (required for persistent-agent paths)

Neither is used anywhere outside the persistent-agent stack, so
existing Hermes users who haven't set them see zero behavior
change.

---

## 10. H6 — [BLOCKED waiting on HIPP0 side]

The brief's Phase H6 ("End-to-end against real HIPP0") cannot be
executed from the Hermes-only working branch because there is no
running HIPP0 instance reachable from the isolated build
environment. The feat branch has been exercised exclusively against
the in-process aiohttp mock (`tests/fixtures/mock_hipp0.py`, which
runs on `127.0.0.1:<random>` and implements all seven endpoints
with failure injection).

**To run H6 manually once a HIPP0 instance is up**:

```bash
export HIPP0_BASE_URL=http://localhost:3000
export HIPP0_API_KEY=…
python -m hermes_cli.agent_registry list      # expect alice, bob
python -c "import asyncio; from tools.persistent_delegate_tool \
  import PersistentDelegateTool; \
  print(asyncio.run(PersistentDelegateTool().invoke('alice', \
  'draft Q3 kickoff')))"
```

Any contract drift discovered during that run should be appended
below this line as new [REQUEST] entries, and the PR description
updated so the HIPP0 instance can address them before the merge.

---

## Deviations from the task brief worth calling out

1. **Upstream path for Hipp0MemoryProvider**: the brief asks for
   `agent/hipp0_memory_provider.py`. Upstream Hermes keeps external
   memory providers under `plugins/memory/<name>/` (see
   `plugins/memory/honcho/`). Hermes went with the brief's path
   because this integration is *in-repo* and not a pluggable
   provider (the brief says: "Do NOT build a Python package called
   hipp0-hermes or publish to PyPI. The integration is in-repo.").
   The file also implements the `MemoryProvider` ABC so the
   existing `MemoryManager` can host it if a future phase wants.

2. **Tests location**: the brief says
   `hermes_cli/tests/test_agent_registry.py`. Upstream Hermes keeps
   all tests under `tests/` (e.g. `tests/hermes_cli/…`), so Hermes
   put the new test file at `tests/hermes_cli/test_agent_registry.py`
   to match the existing conftest.py's HERMES_HOME isolation
   fixture.

3. **Mock server**: aiohttp, as the brief asked, at
   `tests/fixtures/mock_hipp0.py`. No ASGI runner, no separate
   process — it's an in-process TCP server the tests start via an
   `async with start_mock_hipp0()` context manager.

4. **Phased execution**: Hermes worked through H1 → H5 in a single
   authorized session with a commit per phase (see `git log
   feat/persistent-agents-hipp0`). CLAUDE.md's phase-gate rule was
   relaxed for this workstream per the original task author.

5. **Agent name regex**: brief says `^[a-z][a-z0-9_-]{0,63}$`;
   profiles.py uses `^[a-z0-9][a-z0-9_-]{0,63}$` (allows leading
   digit). Agents stay on the brief's stricter form so Telegram
   @mentions and CLI positional args parse cleanly.
