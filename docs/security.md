# Threat model

Scope: the retell-hermes-voice adapter and the trust boundaries it sits on. Wire-level
evidence for every "verified"/"measured" claim below is in
[`integration-contracts.md`](integration-contracts.md) (cited by section).

## Assets

- **The Hermes agent itself** — a high-privilege backend. A default Hermes install
  exposes `terminal`, `process`, `write_file`, `patch`, `execute_code`,
  `browser_exec`, `cronjob`, `delegate_task`, `memory`, and home-automation control
  to any prompt that reaches the API server (§2.8). Whoever can talk to it can
  potentially act as the operator.
- **Caller data** — phone numbers, spoken content, transcripts.
- **Operator data** — anything in Hermes sessions, long-term memory, and the
  operator's machine.
- **Secrets** — `RHV_HERMES_API_KEY`, `RHV_ROUTE_SECRET`, the Retell workspace key.
- **Availability** — the phone line staying answerable.

## Trust boundaries

```
 UNTRUSTED                semi-trusted transport        enforcement point         high privilege
+-----------+  speech   +------------------------+   +---------------------+   +----------------+
|  Caller   | --------> |  Retell cloud          |-->| retell-hermes-voice |-->|  Hermes Agent  |
| (anyone   |           |  STT text of whatever  |WSS|  route secret       |   |  tools, memory |
|  who dials|           |  the caller said;      |   |  allowlist          |   |  operator env  |
|  the      |           |  NO auth on the WS it  |   |  sanitization       |   |                |
|  number)  |           |  opens to the adapter  |   |  session scoping    |   |                |
+-----------+           +------------------------+   |  run lifecycle      |   +----------------+
                                                     +---------------------+
```

- **Caller speech is attacker-controlled input.** Everything in `transcript` frames
  is untrusted, including the literal `(unintelligible audio)` marker.
- **Retell is a semi-trusted transport**: it faithfully carries frames but
  authenticates nothing on the Custom LLM WebSocket — no headers, no signature
  (§1.2). Anyone who knows the URL can impersonate Retell.
- **The adapter is the enforcement point** for everything it can enforce; several
  critical controls (tool restriction, memory scoping) are only enforceable on the
  Hermes side and are called out as such below.
- **Hermes is a high-privilege backend** reached with a bearer key that grants full
  agent access.

## Threats and mitigations

| Threat | Mitigations | Residual risk |
|---|---|---|
| **Prompt injection via caller speech** — caller talks the agent into abusing tools or revealing data | Voice system prompt instructs plain prose, no tool/prompt/error disclosure; `ToolPolicy` defaults to zero enabled tools; **real control**: dedicated least-privilege Hermes profile with dangerous toolsets disabled via `hermes tools disable --platform api_server` | Prompt-level controls are model compliance, not enforcement (§2.8: 8/8 client-side restriction attempts silently ignored). Without the hardened profile, injection = tool access |
| **Prompt injection via tool output / retrieved content** — a tool result contains instructions | Instructions tell the model to treat tool output as data, not instructions; the adapter executes no tools itself | Hermes-side risk: the adapter cannot inspect or filter tool output. Documented, not solved here |
| **Data leakage in logs / spoken output** | `RedactionFilter` scrubs secret values and E.164 numbers from all logs; transcript content is never logged; Hermes fail-open error bodies (HTTP 200 + error prose + zero usage, §2.4) are intercepted and replaced with a generic apology — provider/internal error text is never spoken or logged unredacted | Operators who capture transcripts in their own tooling accept that exposure in their log pipeline |
| **Cross-caller data leakage via Hermes memory** — **VERIFIED in Hermes 0.20.4** (§2.7): a fact planted with "remember this" phrasing in session A was recalled verbatim in a fresh session B, 3 consecutive reproductions; session *keys* did not partition it | Per-call random `session_id`/`session_key` never derived from phone numbers; `session_retention="none"` default | **Headers do not contain this leak.** Operators MUST disable the `memory`/`session_search` toolsets in the voice profile, or explicitly accept that caller A's data can surface in caller B's call |
| **Runaway / abusive tool use** | Advisory budgets `max_tool_calls_per_turn` (3) and `max_tool_seconds_per_call` (60); turn timeout hard-stops the run | Budgets are advisory prompt-shaping. The hard control is the Hermes profile's toolset config. **Do not rely on Hermes approvals**: they are off by default (§2.9), and a pending approval mid-call is silent dead air anyway |
| **DoS / abuse of the endpoint** | Route secret (≥16 chars, constant-time compare, mismatch closed 1008 before processing); `max_concurrent_calls` cap (full → spoken busy line + `end_call`); `max_ws_message_bytes` cap checked before JSON parse; malformed frames ignored, 3 consecutive → close 1008; transcript truncated at `max_transcript_utterances` | Volumetric attacks are handled at the edge (reverse proxy, optional allowlist of Retell's outbound IP 100.20.5.228), not by the adapter |
| **Stale responses spoken to the wrong turn** — Retell accepts content only for the latest `response_id` and drops older ids silently (§1.6) | Session tracks `latest_response_id`, ignores stale requests, cancels the superseded turn task, and re-checks staleness before every send; every cancel stops the Hermes run | A superseded turn may already have executed a Hermes tool — double-invocation is inherent to the protocol (§1.6); idempotency belongs in Hermes-side tools |
| **Orphaned Hermes runs** — abandoning the SSE stream leaves the run alive **unboundedly** (§2.1, measured: still running with live child processes at +30 s) | `POST /v1/runs/{id}/stop` on every barge-in, cancellation, and hangup — measured 0.255 s to `cancelled` vs a ~31 s orphan window for TCP-abort (and unbounded for SSE abandonment); bounded by `hermes_stop_timeout` | None significant; `/stop` on an already-finished run 404s harmlessly |
| **Phone-number privacy** | E.164 numbers redacted in logs (`+1******1234`); call ids logged only as truncated SHA-256 refs; Hermes session ids/keys never derived from phone numbers; retention default `none` | `session_retention="hermes"` persists sessions under Hermes's own retention — an explicit operator choice |

## Deployment-level requirements

These are not adapter features; they are prerequisites for a safe deployment
(details in [`deployment.md`](deployment.md)):

1. WSS termination at a reverse proxy; the adapter binds loopback.
2. A strong route secret; rotate it by updating the Retell agent URL and publishing.
3. A dedicated Hermes voice profile: own `API_SERVER_KEY`, dangerous toolsets
   disabled (including `memory`/`session_search`), approvals irrelevant because the
   toolsets are gone.
4. Optional edge allowlist of Retell's outbound IP `100.20.5.228`.
5. `RHV_ALLOWED_CALLERS` set whenever the caller population is known.
