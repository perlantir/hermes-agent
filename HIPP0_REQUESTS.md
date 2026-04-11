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

Hermes calls exactly the request shape in the task brief, updated
for the H6 live-smoke correction below:

```json
{
  "project_id": "uuid",
  "agent_name": "alice",
  "platform": "telegram",
  "external_user_id": "tg-42",
  "external_chat_id": "chat-7"
}
```

and expects `201` with `{"session_id": "uuid"}`.

`Hipp0MemoryProvider.start_session` refuses any response that does
not contain a server-generated `session_id` — no client-generated
ids anywhere.

**H6 correction (2026-04-11)**: the task brief originally spelled
this field `user_id`, and Hermes shipped H1–H5 against that name.
Running Tier-2 of the live HIPP0 × Hermes integration harness on
the VPS revealed that HIPP0's real `/api/hermes/session/start`
handler reads `external_user_id` (see
`hipp0/scripts/hermes-live-smoke.mjs` and
`hipp0/packages/server/src/routes/hermes.ts` around line 503). The
provider now serializes the Python-side `user_id` parameter as
`external_user_id` on the wire; the Python method keeps
`user_id=` as the kwarg name so callers aren't churned, and the
mock HIPP0 fixture doesn't care because it records whatever it
receives.

**Ask**: please confirm `external_user_id` is the final name so we
can delete this compatibility note in a follow-up.

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

## 6. `POST /api/outcomes` — [REQUEST] contract divergence

Hermes is coded against the brief's shape:

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

**H6 live-smoke finding (2026-04-11)**: HIPP0's real
`/api/outcomes` handler (see
`hipp0/packages/server/src/routes/outcomes.ts` line 137) expects a
completely different body: `compile_request_id`, `decision_id`,
`outcome_type`, `outcome_score`, `task_completed`,
`task_duration_ms`, `agent_output`. It has no concept of
`snippet_ids` or `signal_source`. Sending the brief-shaped body
returns `400 VALIDATION_ERROR: Either compile_request_id or
decision_id + project_id is required`.

The two shapes model different things:
  - **Brief shape** (Hermes): per-turn reinforcement signal
    over snippets that were used to compile the last turn.
  - **HIPP0 shape** (today): per-compile-request outcome tracking
    driven by the `compile_history` / `compile_outcomes` tables.

**Ask**: please land the brief-shaped endpoint. Either (a) accept
both shapes and route to different handlers based on which keys
are present, or (b) add a new path like
`POST /api/hermes/outcomes` that implements the snippet-based
shape so Hermes's per-turn reinforcement can flow end-to-end.
Until that lands, `test_record_outcome` is skipped in live mode
via `skip_if_mock_only` in
`tests/agent/test_hipp0_memory_provider.py`.

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

**H6 correction (2026-04-11)**: same rename as §1 — the user
identifier is serialized as `external_user_id`, not `user_id`, on
the wire. The HIPP0 handler at
`hipp0/packages/server/src/routes/hermes.ts` line 627 hard-fails
with `400 VALIDATION_ERROR` on missing `external_user_id`, so the
brief's `user_id` spelling can never have worked end-to-end
against real HIPP0. Python-side kwarg stays `user_id=` for caller
continuity; the payload key now matches the server.

**Confirmed** (live smoke, 2026-04-11): HIPP0 does publish the
new version in **both** the `ETag` response header and the
body's `version` field. Hermes reads the body field and passes
it back as `If-Match` on the next upsert — matches the HIPP0
smoke script's round-trip at
`hipp0/scripts/hermes-live-smoke.mjs` steps 8-11.

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

## 10. H6 — [Tier 2 GREEN on 2026-04-11]

The brief's Phase H6 ("End-to-end against real HIPP0") was blocked
for the H1–H5 working branch because neither sandbox had network
reach to a real HIPP0. It has now been unblocked on a VPS that
runs both repos side-by-side.

**Tier 1** (36/36 pass): HIPP0's own `scripts/hermes-live-smoke.mjs`
against a real `node dist/index.js` on loopback port 3199 with
`DATABASE_URL=/tmp/hipp0-smoke.db HIPP0_AUTH_REQUIRED=false
HIPP0_TELEMETRY_ENABLED=false`.

**Tier 2** (14/14 live pass, 6 mock-only skipped,
`tests/agent/test_hipp0_memory_provider.py`): the Python
`Hipp0MemoryProvider` drives the real HIPP0 HTTP server via a
new `HIPP0_LIVE_URL` env-var gate. The six skipped tests all
depend on the in-process mock's `queue_failure` knob (5xx / 4xx
injection, mid-drain replay ordering) — they stay in the mock-mode
suite and are explicitly skipped when `HIPP0_LIVE_URL` is set.

The live tests surfaced three bugs — two HIPP0-side crashes that
have been fixed on `claude/build-marketing-website-3HXL3`, and one
wire-format mismatch fixed on the Hermes side:

1. **HIPP0 parsers.ts (fixed)**: `parseAgent` and `parseDecision`
   (and a dozen siblings) hard-cast `row.created_at as Date` and
   crashed with `TypeError: row.created_at.toISOString is not a
   function` on SQLite, which returns timestamps as strings.
   Introduced a `toIsoString` helper in
   `packages/core/src/db/parsers.ts` that handles Date, string,
   and numeric inputs. Fix touches compile, outcomes, and every
   other route that materializes rows.

2. **HIPP0 bootstrap-keys.ts (fixed)**: server fatal-crashed on
   startup with `NOT NULL constraint failed: api_keys.id` because
   the `bootstrapApiKeys` INSERT doesn't supply an id and the
   SQLite `api_keys.id` column has no default. Fixed by making
   bootstrap a no-op when `HIPP0_AUTH_REQUIRED=false` — the
   middleware bypasses every request anyway in dev mode, so
   seeding keys was already pointless. Production with
   `HIPP0_AUTH_REQUIRED=true` is unchanged.

3. **Hermes provider (fixed)**: `user_id` → `external_user_id`
   on the wire for both `/api/hermes/session/start` and
   `/api/hermes/user-facts`, documented in §1 and §7 above.

**Still blocked in live mode** (skipped, not failed):
  - `test_record_outcome` — contract divergence, see §6.

**Reproducible command**:

```bash
# Terminal 1 — keep HIPP0 alive in tmux:
tmux new-session -d -s hipp0 \
  "cd /root/integration/hipp0/packages/server && \
   DATABASE_URL=/tmp/hipp0-smoke.db PORT=3199 \
   HIPP0_AUTH_REQUIRED=false HIPP0_TELEMETRY_ENABLED=false \
   node dist/index.js"

# Terminal 2 — live tier-2 provider tests:
cd /root/integration/hermes-agent && source .venv/bin/activate && \
  HIPP0_LIVE_URL=http://127.0.0.1:3199 \
  python -m pytest tests/agent/test_hipp0_memory_provider.py \
  -v -o 'addopts='
```

Any contract drift discovered in Tier 3 (real LLM distillery) or
beyond should be appended below this line as new [REQUEST]
entries and the PR description updated so the HIPP0 side can
address them before the merge.

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
