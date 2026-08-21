# Integration contracts: Retell AI Custom LLM ↔ Hermes Agent

Canonical distillation of the recon evidence this adapter is built against. Every claim
below carries its verification class:

- **DOCS** — read from the official Retell documentation, spec revision
  `x-retell-spec-revision: 2026-08-12-9e090f0` (stamped on every OpenAPI block fetched).
- **LIVE** — observed by a probe (real request + captured response) against the
  operator's Hermes Agent install or the Retell management API.
- **SOURCE** — read from the installed Hermes implementation
  (`gateway/platforms/api_server.py` of the exact running build), not exercised live.
- **UNVERIFIED / ASSUMED** — inferred; collected in §4.

| Verified at | Component | Version | Method |
|---|---|---|---|
| 2026-08-20 | Retell docs + demo repos | spec rev `2026-08-12-9e090f0` | DOCS (full-page fetches) |
| 2026-08-20 | Retell management API | same spec rev | LIVE (read-only, authenticated) |
| 2026-08-20 | Hermes Agent API server | **0.20.4** (aiohttp 3.14.3, Python 3.11) | LIVE probes + SOURCE reads |

The Retell **LLM WebSocket itself was not exercised live** (no call was placed):
§1 is DOCS-class unless noted; §2 is LIVE or SOURCE throughout.

---

## 1. Retell Custom LLM WebSocket contract (DOCS)

### 1.1 URL pattern

The operator configures a **base URL only** (dashboard "Custom LLM URL" or
`response_engine.llm_websocket_url`); Retell appends `/{call_id}` as the final path
segment at connect time. Configure `wss://host/llm-websocket` → Retell dials
`wss://host/llm-websocket/{call_id}`. Trailing slash is normalized. `https:` is mapped
to `wss:`; plain `ws://` works but is unencrypted. The URL supports Retell
dynamic-variable templating (e.g. `?tenant={{tenant_id}}`). One WebSocket per call;
`call_id` is stable for the life of the call, including across reconnects.

### 1.2 Framing and auth

- All frames are **text frames of stringified JSON**. Retell never sends binary; if the
  server sends a binary frame Retell closes with **code 1007**
  (`error_llm_websocket_corrupt_payload`).
- **Retell sends NO auth headers and no signature on this WebSocket.** The only
  documented protections, which combine: (a) allowlist Retell's single outbound IP
  **100.20.5.228**; (b) embed an unguessable secret in the configured URL (path segment
  or query) and reject connections without it. No HMAC/bearer/signed-URL mechanism
  exists for this endpoint. (The separate REST call-events webhook *does* carry
  `X-Retell-Signature`; unrelated to this socket.)
- Closing the socket with **code 1000 from the server is a deliberate hangup**: the call
  ends as `agent_hangup`, no error, **no reconnect**. Never close 1000 to signal a fault.

### 1.3 Inbound events (Retell → server), discriminator `interaction_type`

Exactly five documented values:

| `interaction_type` | Reply required | Fields (beyond the discriminator) |
|---|---|---|
| `response_required` | yes — `response` | `response_id: int` (required, auto-incrementing), `transcript: Utterance[]` (required, full call transcript so far), `transcript_with_tool_calls?: object[]` |
| `reminder_required` | yes — `response` | same shape as `response_required`; caller went quiet, Retell wants a nudge |
| `update_only` | no | `transcript` (required), `transcript_with_tool_calls?`, `turntaking?: "agent_turn" \| "user_turn"` |
| `call_details` | no | `call: object` — the full call object; **only sent if requested via config** |
| `ping_pong` | yes — echo `ping_pong` | `timestamp: int` (ms epoch); **only sent if `auto_reconnect` set** |

Utterance wire shape: `{role: "agent" | "user", content: str}`; live samples also carry
a `words` array (`{word, start, end}` in float seconds) not in the formal field table.
The official demo repo additionally types `role: "system"` — see §4. Useful
`call_details.call` sub-fields: `call_id`, `agent_id`, `call_type`, `direction`,
`from_number`/`to_number` (phone calls only; absent on web calls),
`retell_llm_dynamic_variables`, `metadata`.

**Magic content value:** when the caller audibly spoke but nothing transcribed
(noise/cough/too quiet), the utterance content is the literal string
`(unintelligible audio)` — not an empty string. Prompt-building must handle it.

### 1.4 Connection open sequence (server obligations)

1. *(optional, but first if sent)* `config` event — Retell acts on it as it arrives:
   `{"response_type": "config", "config": {"auto_reconnect"?: bool, "call_details"?: bool, "transcript_with_tool_calls"?: bool}}`
   Skipping it means: no keepalives, no call_details, no tool-call transcripts.
2. **Begin message**: a `response` event with `response_id: 0` — the first thing the
   agent says. `content: ""` makes the agent wait for the caller to speak first. If the
   greeting depends on call data, send `config` immediately but hold the begin message
   until `call_details` arrives.
3. If `config.call_details = true`, Retell then pushes `call_details` right away.

### 1.5 Outbound `response` event (server → Retell)

```
response_type: "response"          required
response_id: integer               required  — which requested response this answers
content: string                    required  — partial or full content
content_complete: boolean          required  — true only on the LAST event of a response
no_interruption_allowed: boolean   optional  — caller cannot interrupt this content
end_call: boolean                  optional  — hang up after this content is fully spoken
transfer_number: string            optional  — cold transfer after content spoken
show_transferee_as_caller: bool    optional  (default false)
digit_to_press: string             optional  — DTMF after content spoken
```

- `end_call` / `transfer_number` / `digit_to_press` are **mutually exclusive**; Retell
  runs at most one per response_id (`end_call` wins, then `transfer_number`).
- **If the caller interrupts before the content finishes speaking, the attached action
  (hangup/transfer/DTMF) is DISCARDED.**
- A message with no `response_type` is treated as a `response` (backward-compat
  fallback) — always set it explicitly.
- **Silent validation failure:** Retell rejects a response whose `content_complete`
  isn't a boolean / `content` isn't a string / `response_id` isn't an integer — it logs
  server-side, keeps the connection open, and the event just vanishes.

Other outbound `response_type`s: `ping_pong` (echo), `agent_interrupt` (keyed by
`interrupt_id` instead of `response_id`; docs recommend `no_interruption_allowed: true`),
`tool_call_invocation` / `tool_call_result` (§1.8), `update_agent` (mid-call changes to
responsiveness / interruption_sensitivity / reminder settings), `metadata` (web-call
frontend forwarding; see §4).

### 1.6 `response_id` staleness semantics (latest-only, silent drop)

- `response_id` is auto-incrementing per call. **Retell accepts content only for the
  most recently requested id, and only until `content_complete: true` is sent for it.**
- Content under an **older id is dropped silently** — no error frame, no close, no ack.
  Likewise content sent after the id was completed is a no-op.
- **There is no cancel/ack signal.** The server must track the newest id it has seen and
  self-check staleness before every send (docs' own pattern:
  `isStale = request.response_id !== latestResponseId`).
- `update_only` events do **not** supersede an outstanding `response_required` — only a
  newer `response_required`/`reminder_required` with a higher id does.
- Discard-and-retry is normal call behavior (caller kept talking), not an error.
- **Double-tool-invocation risk (DOCS, explicit):** because a superseded response can
  already have executed a tool before the newer request arrives, a tool can run **twice
  for one logical caller turn** and "no setting turns it off". Mitigation is
  server-side idempotency keyed on `call_id` + tool arguments; checking `response_id`
  first helps only when the newer request already arrived.

### 1.7 Keepalive and reconnection

- With `auto_reconnect: true`: Retell sends `ping_pong` **every 2 s**; **5 s without a
  reply → Retell closes and reconnects**. Echo the inbound `timestamp` verbatim (what
  every reference implementation does; wording ambiguity in §4). A blocking response
  handler is the classic cause of missed windows — handle frames concurrently.
- **Initial connection** (socket never opened): up to **3 attempts, 7 s timeout each,
  3 s apart**; all fail → call ends `error_llm_websocket_open`.
- **Mid-call**: up to **2 reconnects** on keepalive loss (then
  `error_llm_websocket_lost_connection`); up to **4 reconnects** on abnormal close
  (WS code 1006; gating unverified, §4).
- Each reconnect is a brand-new handshake to the same URL + same `/{call_id}`. Per-call
  state must be reconstructable from `call_id` alone; every
  `response_required`/`update_only` carries the full transcript, so context survives a
  total server-side state loss. Config must be re-sent per socket (ASSUMED, §4).
- No per-turn response deadline exists: an unanswered `response_required` leaves the
  agent silent until the caller speaks again and triggers a fresh request. The only hard
  connection timing is the 5 s ping/pong deadline.

### 1.8 Tool transcript events

For a custom-LLM agent, **tools are never declared to Retell**; they live entirely in
the server's own LLM stack. Retell's involvement: (a) executing the fixed native actions
attached to a `response` (`end_call`, `transfer_number`, `digit_to_press`); (b)
optionally recording the server's tool calls in the official transcript via:

```
{"response_type": "tool_call_invocation", "tool_call_id": "<globally unique>", "name": "<fn>", "arguments": "<stringified JSON>"}
{"response_type": "tool_call_result", "tool_call_id": "<same id>", "content": "<result>", "successful"?: bool}
```

Optional but valuable: populates `transcript_with_tool_calls` in the Get Call API (and,
with `config.transcript_with_tool_calls = true`, in every inbound event). Omitting
`successful` records "outcome not reported", not failure. Recommended in-turn pattern:
stream a holding line (`content_complete: false`, same `response_id`) → invocation →
run tool → result → continue streaming → close with `content_complete: true`.

---

## 2. Hermes API server contract as installed (v0.20.4, LIVE-verified 2026-08-20)

Target: local Hermes Agent 0.20.4, bearer auth (`Authorization: Bearer <API_SERVER_KEY>`).
Unauthenticated/wrong-key → **401** with an identical body either way
(`type: "gateway_auth_error"`). `GET /health` is **public**:
`{"status": "ok", "platform": "hermes-agent", "version": "0.20.4"}` (LIVE).
`GET /health/detailed` requires auth; its `background_queues.active_api_runs` counts
`/v1/runs` work only (LIVE: stayed 0 through a live streaming chat-completions turn).

### 2.1 Transport decision: `POST /v1/runs` + SSE + mandatory `/stop`

The adapter drives a phone turn with `POST /v1/runs` → `GET /v1/runs/{id}/events` (SSE)
→ `POST /v1/runs/{id}/stop` on any barge-in/hangup/abandon. Evidence (all LIVE):

| Mechanism | Measured stop latency | Orphan risk |
|---|---|---|
| `POST /v1/runs/{id}/stop` | 200 in 0.002 s; status `stopping` → **`cancelled` in 0.255 s**; child process dead in 1.04 s | none |
| Aborting a `stream:true` `/v1/chat/completions` TCP connection | **30.7–30.8 s** (n=2) until the run dies — server only notices the dead socket on its next write, and the only guaranteed write during silence is the 30 s keepalive | bounded ~31 s, uncontrollable |
| Abandoning the `/v1/runs/{id}/events` SSE stream without `/stop` | run **never stops** — still `running` with live child processes at +30 s (full observation window); the run is a detached task that ends only naturally | **unbounded** |

`/stop` on a live run: `{"run_id": …, "status": "stopping"}`. On a finished run:
**404 `run_not_found`** — agent refs are popped at completion while the status record
survives, so `GET /v1/runs/{id}` succeeding does not imply `/stop` will.

### 2.2 `POST /v1/runs` request/response

**202** (LIVE, 0.003 s) `{"run_id": "run_<32 lowercase hex>", "status": "started"}`.
Fields (SOURCE unless noted): `input` (required; string or message array — missing → 400);
`session_id` (defaults to `run_id` → throwaway session); `instructions` (layered ON TOP
of Hermes's own system prompt, never replacing it — LIVE-confirmed);
`conversation_history` (array of `{role, content}`, validated → 400 otherwise; takes
precedence over `previous_response_id`); `model`/`provider`/`model_options` (§2.6; on
this route a bare `model` without `provider` IS honored, unlike chat-completions).
`X-Hermes-Session-Key` is echoed when sent; **`X-Hermes-Session-Id` is NOT echoed on
`/v1/runs`** (LIVE).

### 2.3 `GET /v1/runs/{id}/events` SSE framing (LIVE)

Headers: `Content-Type: text/event-stream`, `Cache-Control: no-cache`,
`X-Accel-Buffering: no`, chunked. Subscribe may race the POST: the handler waits up to
1.0 s (20×50 ms, SOURCE) for the run to register before 404.

**All frames are bare `data:` frames — no `event:` lines on this stream.** The event
name is the JSON `event` field. Observed verbatim (LIVE):

```
data: {"event": "message.delta",        "run_id": …, "timestamp": …, "delta": "The"}
data: {"event": "tool.started",         "run_id": …, "timestamp": …, "tool": "terminal", "preview": "…"}
data: {"event": "tool.completed",       "run_id": …, "timestamp": …, "tool": "terminal", "duration": 0.169, "error": true}
data: {"event": "reasoning.available",  "run_id": …, "timestamp": …, "text": "…"}
data: {"event": "run.completed",        "run_id": …, "timestamp": …, "output": "…", "usage": {"input_tokens": …, "output_tokens": …, "total_tokens": …}}
data: {"event": "run.cancelled",        "run_id": …, "timestamp": …}
```

Text arrives as token-level `message.delta` frames. Keepalive comment `: keepalive`
after 30 s of silence; terminator `: stream closed` then EOF. SOURCE-only additional
events: `run.failed` (`error` text), `approval.request`/`approval.responded`,
`run.steered`, `subagent.start`/`subagent.complete`. `_thinking`/`subagent.tool`
progress are deliberately not forwarded. Parse defensively; ignore unknown names.

**Non-resumability (LIVE):** the stream is single-subscriber and lossy. Re-subscribing
after a hangup returned HTTP 200 but **zero replayed events** (first byte was the 30 s
keepalive); after teardown re-subscribe is 404 (~1.03 s). The only recovery path is
polling `GET /v1/runs/{id}` (200 in 0.002 s; `status` ∈ `queued|running|stopping|
waiting_for_approval|completed|failed|cancelled`; completed runs carry `output` +
`usage`; cancelled runs carry neither). Retention (SOURCE): SSE buffer TTL 300 s,
terminal status records 3600 s.

### 2.4 Error/fail-open behavior (LIVE)

- Uniform JSON error envelope `{"error": {"message", "type", "param", "code"}}` on 4xx;
  but unknown routes/methods return **plain-text** 404/405.
- **Fail-open hazard 1:** an unknown `provider` returns **HTTP 200 in 0.044 s** with the
  error text as the assistant message ("Provider authentication failed: Unknown
  provider …", leading `⚠️`) and **`usage` all zeros**. A naive adapter would read this
  aloud. Detection: normal finish + `usage.total_tokens == 0` and/or known error prefix.
- **Fail-open hazard 2:** a bad *model* with a valid provider produces **no error at
  all** — it silently falls back to the default route (LIVE: normal answer, zero usage).
  Validate model ids at deploy time; there is no runtime signal.
- On `/v1/runs` a bad provider still returns 202; the failure surfaces only in the run.

### 2.5 Concurrency cap (LIVE)

`max_concurrent_runs: 10` on the probed install. 12 parallel `/v1/runs` → 11×202 + 1×429
(the requester may not consume its own last slot, SOURCE — the cap is not exact). 429
body, identical on both agent endpoints, with header **`Retry-After: 1`**:

```json
{"error": {"message": "Too many concurrent runs (max 10)", "type": "rate_limit_error", "param": null, "code": "rate_limit_exceeded"}}
```

The cap is shared across `/v1/runs` and `/v1/chat/completions` — background work can
429 a live phone call.

### 2.6 Per-request model routing + measured TTFB (LIVE, n=3 per row)

An explicit `provider` is **always honored**, even with `direct_model_requests: false`
(SOURCE: the gate only governs a `model` sent *without* `provider`; LIVE-confirmed both
ways — routed TTFB impossible on the default, and a bare model landed squarely in the
default band). `model_options` accepts `reasoning_effort`, `reasoning: {enabled,
effort}`, `service_tier`, `fast` (SOURCE); unknown effort values are silently ignored.

Time to first content delta, identical one-word prompt, streaming (LIVE):

| Route | median TTFB | min | max |
|---|---|---|---|
| `openrouter` / `google/gemini-3.7-flash` + effort low | **0.605 s** | 0.425 | 4.373 |
| `openrouter` / `openai/gpt-5.4-mini` + low | 0.883 s | 0.612 | 2.145 |
| `openrouter` / `anthropic/claude-haiku-4.5` + low | 1.096 s | 0.474 | 7.850 |
| `anthropic` / `claude-haiku-4-5` + low | 1.176 s | 1.158 | 24.046 |
| **default route, no routing fields** | **2.983 s** | 2.225 | 6.157 |

Best routed option ≈ **5× faster than the default**. **Cold-start caveat (LIVE,
reproducible): the first call per provider was the slowest in every config — up to
24.0 s** (provider-client initialization, not steady state). A warmup call per chosen
route is required before real traffic. Shortening the client system prompt is NOT a
latency lever (LIVE: 0-char vs 3,698-char spread was 0.27 s median, inside noise —
the client prompt layers onto Hermes's resident ~27.6k-token prompt).

Tool latency, default route (LIVE, n=3 per row, median TTFB of speakable content):
**0 tools 1.53 s / 1 tool 4.38 s / 3 tools 10.98 s** — each tool round trip ≈ +2.9 s of
dead air. The first `tool.started`-class event arrives ≈2.5 s median, ~1.9 s before the
first content delta — the earliest available "I'm working on it" hook. A multi-tool turn
cannot meet a phone latency budget.

### 2.7 Session semantics (LIVE)

- `X-Hermes-Session-Id` scopes the **conversation transcript**; `X-Hermes-Session-Key`
  scopes **long-term memory**. Independent; send either/both/neither.
- Transcript isolation held in every observation (same-id recall worked; fresh ids got
  `UNKNOWN`; server persists per-session message history).
- **⚠️ Long-term memory is NOT session-scoped — cross-session leak reproduced.** A fact
  planted in session A with explicit "remember this" phrasing was recalled verbatim in a
  brand-new session B in **3 consecutive reproductions (4 of 6 total attempts)**; the
  `memory`/`session_search` toolsets are enabled and process-wide, and a distinct
  session *key* did not reliably partition it. For a phone product: caller A's data can
  surface in caller B's call. The only structural fix is a dedicated Hermes profile with
  `memory`/`session_search` disabled.
- Session-id validation (SOURCE + LIVE): max 256 chars; reject `\r`,`\n`,`\x00`, `..`,
  `/`, `\` — a raw Retell `call_id` must not be embedded behind a `/`. Spaces, `:`, `-`,
  `.`, `_` are fine. Empty id on chat-completions → server derives `api-<16 hex>` from
  the prompt fingerprint, and **identical prompts collide into one session** — always
  send an explicit id. Session key: max 256, rejects `\r\n\x00`, interior `/` allowed.
- `GET /api/sessions` took **55.3 s** (LIVE) — never on a call path.

### 2.8 Tool restriction is IMPOSSIBLE from the API client (LIVE)

Eight structural attempts against a prompt that reliably triggers the `terminal` tool —
`"tools": []`, `"tool_choice": "none"`, both combined, `"toolsets": []`,
`"allowed_tools": []`, `"disabled_toolsets": [...]`, restriction headers — **all
silently ignored; the tool ran every time** (no 400, no warning). SOURCE: `tools`/
`tool_choice` are read only into a request-dedup fingerprint and never reach tool
resolution. The **system prompt is the only client-side lever and it is advisory**
(it worked in the probe, but it is model compliance, not enforcement). Real enforcement
is server-side only: `platform_toolsets.api_server` in the profile's config
(`hermes tools disable --platform api_server …`) — server-wide per profile, not per
request. The probed default port exposes `terminal`, `process`, `write_file`, `patch`,
`execute_code`, `browser_exec`, `cronjob`, `delegate_task`, `memory`, and home-automation
control to any prompt that reaches it.

### 2.9 Approvals

Approvals are **off** on the probed box (`approvals.mode: 'off'`, LIVE: a recursive
delete executed unattended with no `approval.request`). SOURCE surface if ever enabled:
`approval.request` SSE event + status `waiting_for_approval`, answered via
`POST /v1/runs/{id}/approval` `{"choice": "once|session|always|deny"}` (LIVE error
shapes: 400 invalid choice, 409 not pending/not active, 404 unknown run). A blocked
approval is silent dead air — unusable mid-call; keep approvals off for the voice
profile and remove dangerous toolsets instead.

Body-size limits (LIVE): ~5 MB accepted; ~11 MB → **413** in 0.013 s
(`MAX_REQUEST_BYTES = 10_000_000`, SOURCE).

---

## 3. Retell account/management API (spec rev 2026-08-12-9e090f0)

Base `https://api.retellai.com`, bearer auth (401 body:
`{"status":"error","message":"API key is missing or invalid."}`). Routes below are DOCS;
the list/concurrency calls were exercised LIVE (read-only) against the operator's
workspace, which currently contains zero agents and zero phone numbers.

| Purpose | Method + path | Notes |
|---|---|---|
| List agents (voice+chat) | `POST /v2/list-agents` | paginated `{items, has_more, pagination_key}`; filter `filter_criteria.channel.value` ∈ `voice|chat`; there is **no** `/v2/list-chat-agents` (LIVE 404) |
| Get agent | `GET /get-agent/{agent_id}` | `?version=` int, `latest`, `latest_published`, or tag |
| Create agent | `POST /create-agent` | requires `response_engine` + `voice_id`; 201 + AgentResponse |
| Update agent | `PATCH /update-agent/{agent_id}` | **edits the latest DRAFT only** — live/published calls are unaffected until publish; 412 = stale draft (re-GET and retry) |
| Publish draft | `POST /publish-agent/{agent_id}` | LIVE 200 (empty body accepted, `{}`). NOTE: `POST /publish-agent-version` — named in older docs — returns `Cannot POST` and does not exist |
| List phone numbers | `GET /v2/list-phone-numbers` | LIVE 200 |
| Create number | `POST /create-phone-number` | `area_code`, `number_provider` (default `twilio`), `inbound_agents`/`outbound_agents` arrays of `{agent_id, agent_version?, weight}` (weights sum to 1) |
| Repoint number | `PATCH /update-phone-number/{e164}` | send only changed fields; inbound/outbound bindings independent; `agent_version` accepts int, `latest`, `latest_published`, or tag |
| List calls | `POST /v3/list-calls` | v3 is current; transcript/recording via `GET /v1/get-call/{call_id}` |
| Concurrency | `GET /get-concurrency` | LIVE, see below |

**Custom-LLM `response_engine`** (exact, `oneOf` branch `ResponseEngineCustomLm` —
`required: [type, llm_websocket_url]`, no other fields exist on this schema):

```json
{"response_engine": {"type": "custom-llm", "llm_websocket_url": "wss://host.example.com/llm-websocket"}, "voice_id": "<voice>", "agent_name": "…", "language": "en-US"}
```

**Draft-vs-published nuance:** `PATCH /update-agent` + immediately `POST
/publish-agent/{agent_id}` is required for a websocket-URL change to take effect for phone
traffic routed to a published version.

**Deprecated phone-number fields (LIVE):** `create-phone-number` with the singular
`inbound_agent_id` / `outbound_agent_id` fields now returns HTTP 400 `Deprecated API usage
is no longer supported`. Use the plural arrays, and note `weight` is REQUIRED on each entry
(omitting it returns 400 `must have required property 'weight'`).

**Buying a number requires billing (LIVE):** `POST /create-phone-number` on a workspace with
no payment method returns HTTP 402 `This item requires a card on file` after passing
validation — provision billing before automating number purchase.

**Concurrency (LIVE):** `base_concurrency: 20`, `concurrency_burst_enabled: true`,
`concurrency_burst_limit: 60` (burst = min(3×limit, limit+300); calls 21–60 proceed at a
$0.10/min surcharge). Inbound calls queue ~40 s for a free slot before `fallback_number`
or `concurrency_limit_reached` (DOCS). Max call duration default 1 h
(`max_call_duration_ms` up to 2 h).

Agent-level fields that matter for the bridge (all top-level on AgentRequest, DOCS):
`interruption_sensitivity`, `responsiveness`, `reminder_trigger_ms`/`reminder_max_count`
(drives `reminder_required`), `begin_message_delay_ms`, `end_call_after_silence_ms`
(default 10 min), `webhook_url`/`webhook_events` (agent-level webhook fully **replaces**
the account-level one; the account-level webhook and the webhook signing key are
dashboard-only, not settable via API). Custom-LLM agents cannot use LLM
Playground/simulation testing (Web Call and Phone Call only) and cannot be the target of
an Agent Transfer node.

---

## 4. Assumed / unverified

Adapter-relevant items explicitly flagged in the recon, none confirmed live:

- **Max WS message/payload size on the Retell socket: undocumented anywhere.** The
  adapter enforces its own inbound cap and treats oversized frames as malformed.
- **1006-reconnect gating:** the 4-reconnect abnormal-close path is documented separately
  from `auto_reconnect`; whether it requires the flag is UNVERIFIED. Only the ping/pong
  2-reconnect path is explicitly tied to `auto_reconnect: true`.
- **Outbound `ping_pong` timestamp wording:** the spec text says "when YOUR SERVER sent
  this event", but every official example echoes Retell's inbound timestamp verbatim.
  Echoing is the behavior to follow; sending own clock is plausible but unverified.
- **Config re-send on reconnect:** ASSUMED required (config is per-socket, and a
  reconnect is a fresh socket); not explicitly documented end-to-end.
- **Demo-repo `role: "system"` utterance value:** appears only in the possibly-stale
  official demo's types, not in current docs. Tolerated on parse, dropped for Hermes.
- **`response_required` sample `timestamp` field:** in the docs' sample JSON but absent
  from the formal field table (which the sample also contradicts by omitting
  `response_id`). Field table is authoritative; ignore stray `timestamp`.
- **`metadata` outbound event forwarding:** cross-referenced to the audio-websocket docs
  page, not fetched. Irrelevant to phone calls.
- **Hermes non-`stop` finish shapes** (`finish_reason: length|error`, `hermes:{…}`
  envelope, `run.failed` payload): SOURCE-read, never reproduced live.
- **Hermes dedicated-profile commands** (§2.8 enforcement): command surface read from
  `--help`/shipped docs; nothing was executed, and the exact `platform_toolsets.api_server`
  key write was not confirmed by running it. Verify with `GET /v1/toolsets` after applying.
- **No live Retell WS session was captured** — §1 is DOCS-class in its entirety; wire
  behavior (e.g. exact `words` payloads, real close sequences) is untested against a
  real call.

---

## 5. Consequences for this adapter (finding → design decision)

- **No auth on the Retell WS** → route is `WS /llm-websocket/{route_secret}/{call_id}`
  with a ≥16-char unguessable secret compared via `secrets.compare_digest`; mismatch
  closes 1008 before processing. Deployment docs additionally recommend allowlisting
  Retell's outbound IP `100.20.5.228` at the edge.
- **Abandoned runs SSE never stops the run (LIVE, unbounded)** → `HermesClient.run_turn`
  ALWAYS issues `POST /v1/runs/{id}/stop` on generator close/cancellation, bounded by
  `hermes_stop_timeout`. `/stop` is the only 0.255 s barge-in mechanism; chat-completions
  is rejected as a transport outright (~31 s uncancellable orphan).
- **Cross-session memory leak (LIVE, 3/3 with "remember" phrasing)** → per-call random,
  non-phone-derived `session_id`/`session_key` (`derive_session_ids`), default
  `session_retention: "none"`; deployment requires a dedicated Hermes voice profile with
  `memory`/`session_search` disabled — headers alone do not contain the leak.
- **Cold-start TTFB up to 24 s per provider (LIVE)** → `warmup_on_start: True` fires one
  tiny routed run at startup; `/readyz` gates traffic on Hermes health.
- **Default route 2.98 s vs routed 0.605 s median (LIVE)** → `voice_model` /
  `voice_provider` / `voice_reasoning_effort: "low"` injected on every run; provider is
  always sent with model (bare model is ignored on chat-completions and silently
  mis-falls-back elsewhere).
- **Tool turns cost +2.9 s each; first tool event ≈2 s before first text (LIVE)** →
  filler phrases: a `filler_after_seconds` dead-air timer races the first delta and is
  re-armed on `tool_start` events, speaking a non-repeating holding line.
- **Hermes fail-open errors return HTTP 200 with error prose + zero usage (LIVE)** →
  `run_turn` detects error-as-content (zero usage + known prefixes) and yields a safe
  generic `kind="error"`; the session speaks one apology sentence — provider/internal
  error text is never spoken or logged unredacted.
- **Client-side tool restriction impossible (LIVE, 8/8 ignored)** → `ToolPolicy` is
  honestly documented as advisory prompt-shaping only; real enforcement is the dedicated
  Hermes profile's `platform_toolsets` (deployment requirement, not adapter code).
- **Latest-only response_id with silent drops (DOCS)** → `CallSession` tracks
  `latest_response_id`, ignores stale requests, cancels the prior turn task (which stops
  its Hermes run), and re-checks staleness before every send.
- **Double tool invocation across discarded responses (DOCS)** → voice instructions
  budget tool calls per turn; true idempotency keying (call_id + args) belongs to
  Hermes-side tools since the adapter executes none itself.
- **`(unintelligible audio)` magic content (DOCS)** → passed through to Hermes unchanged
  with instructions to infer intent or ask for a repeat.
- **Binary frames close 1007; malformed responses vanish silently (DOCS)** → adapter
  sends text-only, validates outbound models via pydantic (`dump_outbound`), enforces an
  inbound size cap before JSON parse, ignores malformed frames (closing 1008 after 3
  consecutive), and never closes 1000 except for a deliberate hangup.
- **Reconnects are fresh sockets to the same call_id (DOCS)** → config + greeting logic
  run per-connection; per-call state is reconstructable from the transcript carried on
  every inbound event.
- **429 at `max_concurrent_runs` with `Retry-After: 1` (LIVE)** → adapter caps its own
  `max_concurrent_calls` below the Hermes cap and speaks a busy line + `end_call` when
  saturated, rather than surfacing a rate-limit error mid-call.
- **SSE non-resumable (LIVE)** → a dropped events stream is treated as a failed turn
  (stop the run, apologize); recovery-by-poll (`GET /v1/runs/{id}`) exists but replayed
  audio is impossible, so the adapter does not attempt resubscription.
