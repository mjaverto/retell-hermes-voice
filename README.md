> [!WARNING]
> **DEPRECATED — this project is no longer maintained.**
> It has been replaced by **[vapi-hermes-voice](https://github.com/mjaverto/vapi-hermes-voice)**,
> the same Hermes voice bridge built on [Vapi](https://vapi.ai) instead of Retell
> (voice-provider change, not a technical failure of this adapter). The Hermes-side
> engineering — run lifecycle, speech sanitization, security posture, and the
> verified Hermes wire contract in `docs/integration-contracts.md` — carries forward
> there. This repository is archived and read-only.

# retell-hermes-voice

A Python 3.11 FastAPI adapter that bridges the [Retell AI](https://retellai.com)
Custom LLM WebSocket protocol to a Hermes Agent API server, so a phone caller can
talk to your Hermes agent.

```
                 PSTN / SIP                WebSocket (Custom LLM)          HTTP + SSE
 +--------+    +-----------------+       +----------------------+       +---------------+
 | Caller | -> | Retell AI cloud | <---> | retell-hermes-voice  | <---> | Hermes Agent  |
 +--------+    |  telephony      |  WSS  |  (this adapter)      |       |  API server   |
               |  STT / TTS      |       |  /llm-websocket/     |       |  /v1/runs     |
               |  turn-taking    |       |    {secret}/{call}   |       |  SSE events   |
               +-----------------+       +----------------------+       +---------------+
```

## Division of labor

| Layer | Responsibility |
|---|---|
| **Retell** | Telephony (numbers, SIP), speech-to-text, text-to-speech, turn-taking, barge-in detection, call recording |
| **This adapter** | Protocol translation, route-secret auth, caller allowlist, latency management (fillers, model routing, warmup), speech sanitization, stale-response guarding, run lifecycle (no orphaned Hermes runs), redacted logging |
| **Hermes Agent** | Reasoning, tools, memory, sessions — the actual "brain" |

### Non-goals

- Not a SIP stack or telephony system — Retell owns the phone leg.
- Not a Hermes fork or plugin — it is a pure API client of the Hermes API server.
- Does not make every Hermes tool voice-safe. Tool restriction is enforced on the
  Hermes side (see [Security](#security)); the adapter's tool policy is advisory.

## Supported versions

- Python **>= 3.11**
- Hermes Agent **0.20.4** (verified live)
- Retell Custom LLM protocol, spec revision **2026-08-12** (verified against docs)

The full verified wire contract — every frame shape, latency number, and failure mode
this adapter is built against — lives in
[`docs/integration-contracts.md`](docs/integration-contracts.md).

## Quick start

```sh
# 1. Install
uv venv
uv pip install -e '.[dev]'

# 2. Configure
cp .env.example .env
# edit .env: set RHV_HERMES_BASE_URL, RHV_HERMES_API_KEY, RHV_ROUTE_SECRET

# 3. Enable the Hermes API server (on the Hermes host)
#    In ~/.hermes/.env:
#      API_SERVER_ENABLED=true
#      API_SERVER_KEY=<a strong secret>       # this becomes RHV_HERMES_API_KEY
#    then start/restart the gateway:
hermes gateway

# 4. Run the adapter
python -m retell_hermes_voice

# 5. Verify
curl http://127.0.0.1:8765/healthz   # {"status": "ok"}
curl http://127.0.0.1:8765/readyz    # 200 when Hermes is reachable, 503 otherwise
```

Generate a route secret (minimum 16 characters enforced; use 32+):

```sh
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

## Exposing the WebSocket (local tunnel)

Retell must reach your adapter over `wss://`. For development, tunnel the local port:

```sh
# cloudflared (also a legitimate production option — see docs/deployment.md)
cloudflared tunnel --url http://127.0.0.1:8765

# ngrok (development only)
ngrok http 8765
```

The URL you paste into Retell (dashboard "Custom LLM URL" or the
`response_engine.llm_websocket_url` API field) is:

```
wss://<your-host>/llm-websocket/<your-route-secret>
```

Retell appends `/{call_id}` as the final path segment at connect time, producing the
adapter's actual route `WS /llm-websocket/{route_secret}/{call_id}`
(see `docs/integration-contracts.md` §1.1).

## Configuring the Retell agent

Via the dashboard: create a voice agent, choose **Custom LLM** as the response engine,
paste the `wss://.../llm-websocket/<route_secret>` URL, pick a voice, then **publish**
and bind a phone number.

Via the management API (`https://api.retellai.com`, bearer auth — see
`docs/integration-contracts.md` §3):

```sh
# Create the agent
curl -s https://api.retellai.com/create-agent \
  -H "Authorization: Bearer $RETELL_API_KEY" -H "Content-Type: application/json" \
  -d '{
    "response_engine": {"type": "custom-llm",
      "llm_websocket_url": "wss://your-host.example.com/llm-websocket/<route_secret>"},
    "voice_id": "<voice>", "agent_name": "hermes-voice", "language": "en-US"
  }'

# Publish (PATCH /update-agent edits the DRAFT only; a publish is required
# before phone traffic sees the change)
curl -s -X POST https://api.retellai.com/publish-agent/<agent_id> \
  -H "Authorization: Bearer $RETELL_API_KEY" -H "Content-Type: application/json" \
  -d '{}'

# Bind a phone number
curl -s -X PATCH https://api.retellai.com/update-phone-number/+15551234567 \
  -H "Authorization: Bearer $RETELL_API_KEY" -H "Content-Type: application/json" \
  -d '{"inbound_agents": [{"agent_id": "<agent_id>", "weight": 1}]}'
```

## Configuring Hermes

Enable the API server as in Quick start. Then, **strongly recommended**: run a
dedicated, least-privilege Hermes profile for voice, and disable dangerous toolsets
on the API-server platform:

```sh
hermes tools disable --platform api_server terminal process write_file patch \
  execute_code browser_exec cronjob delegate_task memory session_search
```

This is not optional hardening theater. It was verified live
(`docs/integration-contracts.md` §2.8) that **API clients cannot restrict Hermes
tools** — `"tools": []`, `"tool_choice": "none"`, toolset filters, and restriction
headers are all silently ignored. The adapter's `RHV_TOOL_POLICY__*` settings are
advisory prompt-shaping only. The Hermes profile config is the only real enforcement
point. Disabling `memory`/`session_search` also closes a verified cross-session
memory leak (see [docs/security.md](docs/security.md)).

## Voice behavior

- **Greeting** — inbound calls are answered with `RHV_GREETING` (as Retell
  `response_id` 0). Outbound calls open with `RHV_OUTBOUND_OPENING`, a template that
  states the assistant's identity and a recording disclosure, formatted with
  `RHV_ASSISTANT_NAME`, `RHV_PRINCIPAL`, and an optional `{purpose}` from Retell
  dynamic variables.
- **Filler phrases** — if no speakable text has arrived within
  `RHV_FILLER_AFTER_SECONDS` (default 1.5 s), the adapter speaks a non-repeating
  holding line from `RHV_FILLER_PHRASES`; the timer re-arms on tool-start events,
  since each Hermes tool round trip adds roughly 2.9 s of dead air.
- **Sanitization** — Hermes output is converted to speakable prose before it reaches
  TTS: markdown, code fences, tables, and emoji are stripped; URLs become
  "a link I can send you". Streaming-safe (constructs that span deltas are buffered).

### Latency

Measured against Hermes 0.20.4 (`docs/integration-contracts.md` §2.6):

- Default Hermes route: **~3.0 s** median time-to-first-token. Routed
  `openrouter` / `google/gemini-3.7-flash` at low reasoning effort: **0.605 s**
  median — about 5× faster. Set the routing trio:

  ```sh
  RHV_VOICE_MODEL=google/gemini-3.7-flash
  RHV_VOICE_PROVIDER=openrouter
  RHV_VOICE_REASONING_EFFORT=low
  ```

  Always set provider together with model — a bare model without a provider is
  silently ignored or mis-routed by Hermes.
- **Cold start**: the first call per provider was measured at up to **24 s**.
  `RHV_WARMUP_ON_START=true` (default) fires one tiny routed run at startup so the
  first real caller doesn't pay it. `/readyz` gates traffic on Hermes health.

## Security

> **The WebSocket endpoint is effectively public.** Retell sends **no auth headers
> and no signature** on the Custom LLM socket. Anyone who learns the URL can talk to
> your Hermes agent. You MUST combine:
>
> 1. The unguessable **route secret** in the path (compared constant-time; mismatch
>    is rejected before any event is processed).
> 2. **WSS** termination at a reverse proxy (never expose plain `ws://`).
> 3. Optionally, allowlist Retell's single documented outbound IP **100.20.5.228**
>    at the edge.
>
> Additionally: set `RHV_ALLOWED_CALLERS` to restrict who may reach the agent
> (empty list = allow all, with a warning), and run Hermes with **no tools enabled**
> for the API-server platform unless you have explicitly reviewed each one.

Full threat model: [docs/security.md](docs/security.md).
Reporting vulnerabilities: [SECURITY.md](SECURITY.md).

## Configuration reference

All settings load from `RHV_`-prefixed environment variables or a `.env` file
(case-insensitive). List fields accept comma-separated strings or JSON arrays.

| Env var | Default | Description |
|---|---|---|
| `RHV_HERMES_BASE_URL` | *(required)* | Hermes API server base URL, e.g. `http://127.0.0.1:8642` |
| `RHV_HERMES_API_KEY` | *(required)* | Hermes `API_SERVER_KEY` bearer token (secret) |
| `RHV_ROUTE_SECRET` | *(required)* | Unguessable WebSocket path segment; minimum 16 characters enforced |
| `RHV_LISTEN_HOST` | `127.0.0.1` | Bind address (keep loopback; terminate WSS at a proxy) |
| `RHV_LISTEN_PORT` | `8765` | Bind port |
| `RHV_ALLOWED_CALLERS` | `[]` | E.164 caller allowlist; empty = allowlist disabled (allow all, warn) |
| `RHV_GREETING` | `Hi, this is the assistant. How can I help?` | Inbound-call opening line |
| `RHV_OUTBOUND_OPENING` | identity + recording-disclosure template | Outbound opening; `{assistant_name}`, `{principal}`, `{purpose}` placeholders |
| `RHV_ASSISTANT_NAME` | `the assistant` | Name the assistant introduces itself with |
| `RHV_PRINCIPAL` | `the operator` | Whose assistant it says it is |
| `RHV_FILLER_PHRASES` | 8 built-in phrases | Holding lines spoken during dead air; must be non-empty |
| `RHV_FILLER_AFTER_SECONDS` | `1.5` | Dead-air threshold before a filler is spoken |
| `RHV_VOICE_MODEL` | *(unset)* | Hermes model override for voice turns, e.g. `google/gemini-3.7-flash` |
| `RHV_VOICE_PROVIDER` | *(unset)* | Provider for `RHV_VOICE_MODEL`; always set together with it |
| `RHV_VOICE_REASONING_EFFORT` | `low` | `model_options.reasoning_effort` sent to Hermes |
| `RHV_WARMUP_ON_START` | `true` | Fire one tiny routed run at startup to absorb provider cold start |
| `RHV_HERMES_CONNECT_TIMEOUT` | `5.0` | Hermes HTTP connect timeout (s) |
| `RHV_HERMES_FIRST_TOKEN_TIMEOUT` | `15.0` | Max wait for the first Hermes token (s) |
| `RHV_HERMES_TURN_TIMEOUT` | `60.0` | Max wall time for one Hermes turn (s) |
| `RHV_HERMES_STOP_TIMEOUT` | `3.0` | Bound on the mandatory `POST /v1/runs/{id}/stop` (s) |
| `RHV_MAX_CONCURRENT_CALLS` | `5` | Adapter concurrent-call cap; keep below Hermes `max_concurrent_runs` (default 10) |
| `RHV_MAX_TRANSCRIPT_UTTERANCES` | `200` | Transcript truncation (keeps most recent) |
| `RHV_MAX_WS_MESSAGE_BYTES` | `1000000` | Inbound WS frame size cap, checked before JSON parse |
| `RHV_SESSION_RETENTION` | `none` | `none` = per-call random session ids; `hermes` = let Hermes persist sessions |
| `RHV_TOOL_POLICY__ENABLED_TOOLS` | `[]` | Advisory: tools voice turns may use (prompt-shaping only, not enforcement) |
| `RHV_TOOL_POLICY__CONFIRM_TOOLS` | `[]` | Advisory: tools requiring spoken confirmation |
| `RHV_TOOL_POLICY__MAX_TOOL_CALLS_PER_TURN` | `3` | Advisory per-turn tool budget |
| `RHV_TOOL_POLICY__MAX_TOOL_SECONDS_PER_CALL` | `60.0` | Advisory per-call tool time budget |

## Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| `/readyz` returns 503 | Hermes API server unreachable — check `RHV_HERMES_BASE_URL`, that `API_SERVER_ENABLED=true`, and that `hermes gateway` is running |
| 401 from Hermes | `RHV_HERMES_API_KEY` doesn't match `API_SERVER_KEY` in `~/.hermes/.env` (Hermes returns identical bodies for missing and wrong keys) |
| Retell connects, then silence | Route secret mismatch (connection closed 1008 before events), or the config frame never reached Retell — verify the URL you configured includes the secret and that the proxy forwards WebSocket upgrades on `/llm-websocket` |
| First call is very slow | Provider cold start (up to 24 s measured) — leave `RHV_WARMUP_ON_START=true` and wait for `/readyz` before routing traffic |
| Caller hears "all lines are busy" | Adapter concurrent-call cap reached (`RHV_MAX_CONCURRENT_CALLS`), or Hermes is at `max_concurrent_runs` — note the Hermes cap is shared with all other API work |
| Agent speaks markdown artifacts | Should not happen (sanitizer); if it does, file a bug with the raw Hermes output |

## Testing

```sh
pytest
```

The suite is fully offline: a programmable fake Hermes ASGI app is mounted via
`httpx.ASGITransport`, and no real network is touched. There are no live tests in CI;
any test requiring real Retell or Hermes credentials is opt-in and local-only.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT — see [LICENSE](LICENSE).
