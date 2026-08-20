# Deployment

## Process model

The adapter is a **long-lived WebSocket server** (`python -m retell_hermes_voice`,
uvicorn under the hood). One WebSocket = one phone call, held open for the call's
lifetime; each call fans in a Hermes SSE stream and keeps per-call state
(latest `response_id`, greeted flag, filler picker, in-flight turn task) in memory.

Serverless request handlers are not sufficient for this workload:

- The socket lives as long as the call (minutes), far beyond typical
  function-invocation limits, and Retell's 2 s ping/pong keepalive (5 s deadline)
  tolerates no cold starts mid-call.
- Per-call state is in-memory between frames.
- Each turn holds a second long-lived connection (Hermes SSE) that must be
  cancelled server-side (`/stop`) the instant the caller barges in.

Run it as a supervised process on the same host as (or with low-latency access to)
the Hermes API server.

## systemd unit

On the operator's Hermes host, binding loopback behind a reverse proxy:

```ini
# /etc/systemd/system/retell-hermes-voice.service
[Unit]
Description=Retell <-> Hermes voice adapter
After=network-online.target
Wants=network-online.target

[Service]
Type=exec
User=rhv
Group=rhv
WorkingDirectory=/opt/retell-hermes-voice
ExecStart=/opt/retell-hermes-voice/.venv/bin/python -m retell_hermes_voice
# Secrets live in the environment file, never in the unit itself:
EnvironmentFile=/etc/retell-hermes-voice/env
Restart=on-failure
RestartSec=2
# Hardening
NoNewPrivileges=true
ProtectSystem=strict
ProtectHome=true
PrivateTmp=true

[Install]
WantedBy=multi-user.target
```

`/etc/retell-hermes-voice/env` (mode `0600`, owned by root):

```sh
RHV_HERMES_BASE_URL=http://127.0.0.1:8642
RHV_HERMES_API_KEY=...
RHV_ROUTE_SECRET=...
```

## WSS termination

The adapter listens on plain HTTP/WS at `127.0.0.1:8765`. Terminate TLS at a proxy
and forward WebSocket upgrades on the `/llm-websocket` path.

### Caddy

```caddyfile
voice.example.com {
    # health endpoints for load balancers (optional to expose)
    handle /healthz { reverse_proxy 127.0.0.1:8765 }
    handle /readyz  { reverse_proxy 127.0.0.1:8765 }

    # Caddy proxies WebSocket upgrades automatically
    handle /llm-websocket/* {
        reverse_proxy 127.0.0.1:8765
    }
}
```

### nginx

```nginx
server {
    listen 443 ssl;
    server_name voice.example.com;
    # ssl_certificate / ssl_certificate_key ...

    location /llm-websocket/ {
        proxy_pass http://127.0.0.1:8765;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_read_timeout 3600s;   # calls can last up to Retell's 1-2h max duration
        proxy_send_timeout 3600s;
    }

    location = /healthz { proxy_pass http://127.0.0.1:8765; }
    location = /readyz  { proxy_pass http://127.0.0.1:8765; }
}
```

Optionally restrict `/llm-websocket/` to Retell's single documented outbound IP
`100.20.5.228` (`allow 100.20.5.228; deny all;` in nginx, `remote_ip` matcher in
Caddy). This is defense in depth on top of the route secret, not a replacement.

### Cloudflare Tunnel

`cloudflared` is a legitimate **production** ingress: a persistent, authenticated
outbound tunnel (no inbound firewall holes), TLS handled by Cloudflare, and it
proxies WebSockets. Run it as its own systemd service pointing at
`http://127.0.0.1:8765`. ngrok is fine for development but is not a production
ingress (ephemeral URLs, session limits).

## Health and readiness

- `GET /healthz` — liveness; always `{"status": "ok"}` while the process runs.
- `GET /readyz` — readiness; 200 only when Hermes answers its health check, 503
  otherwise. Point load-balancer/orchestrator readiness probes here so traffic
  (and Retell agent publishing) waits for Hermes — this also ensures the startup
  warmup run has a live backend, avoiding the measured up-to-24 s provider cold
  start on a first real call.

## Graceful shutdown and restarts

On SIGTERM the adapter closes in-flight call sessions — each cancels its turn and
issues the mandatory `POST /v1/runs/{id}/stop` (measured 0.255 s to `cancelled`;
without it an abandoned run keeps running unboundedly) — then closes the shared
Hermes client.

**A restart drops active WebSockets.** This is survivable: per
`integration-contracts.md` §1.7, Retell reconnects to the same URL + `call_id` —
up to 2 times on keepalive loss and up to 4 on abnormal close — and every
`response_required` carries the full call transcript, from which the adapter
rebuilds per-call state on the fresh socket (config frame is re-sent
per-connection). Callers experience a pause, not a dropped call, as long as the
process is back within the reconnect window. Still: prefer draining (stop routing
new calls, let active ones finish) for planned maintenance.

## Resource sizing

The workload is IO-bound: JSON frames in, SSE deltas out, no local inference.
**~1 vCPU and 256 MB RAM** comfortably handles the default
`RHV_MAX_CONCURRENT_CALLS=5`. The real ceiling is Hermes: its
`max_concurrent_runs` (default 10) is shared across **all** API clients, so
background Hermes work can 429 a live phone call. Keep the adapter's cap below the
Hermes cap with headroom for whatever else talks to that Hermes instance.

## Logs and metrics

Logging is structured `key=value` to stderr — under systemd it lands in journald:

```sh
journalctl -u retell-hermes-voice -f
```

Secrets and phone numbers are redacted at the logging layer; transcripts are not
logged unless `RHV_LOG_TRANSCRIPTS=true`. The fields worth graphing are the
per-turn latency figures **`ttfb_ms`** (time to first speakable token — the number
callers feel; routed models measured at 0.605 s median vs ~3.0 s default) and
**`total_ms`** (full turn wall time). Alert on `ttfb_ms` creeping toward
`RHV_FILLER_AFTER_SECONDS` × N and on `/readyz` flapping.

## Retention and backup

By default the adapter persists **nothing**: no database, no transcript files, and
`RHV_SESSION_RETENTION=none` gives every call fresh random Hermes session ids.
There is nothing to back up; back up your `/etc/retell-hermes-voice/env` secrets
through your normal secret-management channel instead.

Setting `RHV_SESSION_RETENTION=hermes` makes Hermes persist per-session
conversation history under its own retention rules — call content then lives on
the Hermes host, and the cross-session memory caveat in
[`security.md`](security.md) applies with more surface. Treat that as an explicit
data-retention decision, not a tuning knob.

## Dedicated Hermes voice profile

Run voice against its own least-privilege Hermes profile, not your daily-driver
agent:

1. Create a separate Hermes profile with its own `~/.hermes/.env`-equivalent:
   distinct `API_SERVER_KEY` and its own API server port.
2. Disable dangerous toolsets for the API-server platform in that profile:

   ```sh
   hermes tools disable --platform api_server terminal process write_file patch \
     execute_code browser_exec cronjob delegate_task memory session_search
   ```

   (Client-side restriction is impossible — verified; and disabling
   `memory`/`session_search` closes the verified cross-session memory leak. Verify
   the result via `GET /v1/toolsets`.)
3. Point the adapter at that profile: `RHV_HERMES_BASE_URL=http://127.0.0.1:<port>`
   with the profile's `API_SERVER_KEY` as `RHV_HERMES_API_KEY`.

This isolates the phone-reachable surface from the operator's primary agent even
if every other control fails.
