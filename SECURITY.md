# Security policy

## Supported versions

| Version | Supported |
|---|---|
| 0.1.x | yes |
| < 0.1 | no |

## Reporting a vulnerability

Report vulnerabilities privately via **GitHub Security Advisories** on this
repository ("Security" tab → "Report a vulnerability"). Do not open a public issue
for anything exploitable.

Please include: affected version/commit, a reproduction (or a clear description of
the attack path), and impact assessment. Redact any real phone numbers, call ids,
API keys, or transcripts from your report.

## Response expectations

- Acknowledgement within 7 days.
- Assessment and severity triage within 14 days.
- Fixes for confirmed vulnerabilities land in a patch release; the advisory is
  published after a fix is available.

This is a small open-source project; there is no bug bounty.

## Scope

In scope: this adapter — the WebSocket endpoint, route-secret handling, caller
allowlist, session-id derivation, log redaction, Hermes run lifecycle, and anything
else in this repository.

Out of scope (report upstream):

- **Hermes Agent** issues (tool enforcement, memory scoping, API server auth) —
  report to the Hermes project. Known Hermes-side risks this adapter documents and
  mitigates around are described in [docs/security.md](docs/security.md).
- **Retell AI** platform issues (telephony, the Custom LLM protocol itself, their
  management API) — report to Retell.

Deployment misconfiguration (e.g. exposing the adapter without WSS or with a weak
route secret) is an operator responsibility; hardening guidance lives in
[docs/deployment.md](docs/deployment.md) and [docs/security.md](docs/security.md).
