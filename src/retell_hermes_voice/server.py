"""FastAPI app factory: health endpoints and the Retell Custom LLM WebSocket."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import logging
import secrets
import time
from collections.abc import AsyncIterator

import httpx
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from .call_session import CallSession
from .config import Settings
from .hermes_client import HermesClient, HermesUnavailableError
from .retell_events import (
    ConfigOut,
    ResponseOut,
    RetellConfig,
    RetellProtocolError,
    dump_outbound,
    parse_inbound,
)

logger = logging.getLogger(__name__)

BUSY_LINE = "I'm sorry, all lines are busy right now. Please call back shortly."
MAX_CONSECUTIVE_MALFORMED = 3


def _call_ref(call_id: str) -> str:
    """Stable, non-reversible log reference for a call id."""
    return hashlib.sha256(call_id.encode()).hexdigest()[:12]


def create_app(
    settings: Settings, hermes_transport: httpx.AsyncBaseTransport | None = None
) -> FastAPI:
    """Build the ASGI app; one HermesClient and one call-slot counter per app.

    ``hermes_transport`` lets tests mount a fake Hermes backend in-process.
    """
    # Plain counter, not a Semaphore: the busy check and the slot grab must be a
    # single atomic step (no await between them), which locked()-then-acquire
    # cannot guarantee. asyncio is single-threaded, so check+increment with no
    # intervening await is race-free.
    active_calls = 0
    config_frame = dump_outbound(
        ConfigOut(config=RetellConfig(auto_reconnect=True, call_details=True))
    )

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        hermes = HermesClient(settings, transport=hermes_transport)
        app.state.hermes = hermes
        app.state.sessions = set()
        warmup_task: asyncio.Task[None] | None = None
        if settings.warmup_on_start:
            warmup_task = asyncio.create_task(hermes.warmup(), name="rhv-warmup")
        try:
            yield
        finally:
            if warmup_task is not None:
                warmup_task.cancel()
                try:
                    await warmup_task
                except asyncio.CancelledError:
                    pass
                except Exception as exc:
                    logger.warning(
                        "warmup cancel failed during shutdown error=%s", type(exc).__name__
                    )
            sessions: set[CallSession] = app.state.sessions
            for session in list(sessions):
                try:
                    await session.close()
                except Exception as exc:
                    logger.warning(
                        "session close failed during shutdown error=%s", type(exc).__name__
                    )
            await hermes.aclose()

    app = FastAPI(lifespan=lifespan)

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz")
    async def readyz() -> JSONResponse:
        hermes: HermesClient = app.state.hermes
        try:
            ready = await hermes.health()
        except HermesUnavailableError as exc:
            logger.warning("readiness check failed error=%s", type(exc).__name__)
            ready = False
        else:
            if not ready:
                logger.warning("readiness check failed error=unhealthy_status")
        if ready:
            return JSONResponse({"status": "ready"})
        return JSONResponse({"status": "degraded"}, status_code=503)

    @app.websocket("/llm-websocket/{route_secret}/{call_id}")
    async def llm_websocket(websocket: WebSocket, route_secret: str, call_id: str) -> None:
        nonlocal active_calls
        if not secrets.compare_digest(
            route_secret.encode(), settings.route_secret.get_secret_value().encode()
        ):
            # Starlette requires accept before close; no events are processed.
            await websocket.accept()
            await websocket.close(code=1008)
            logger.warning("ws rejected call=%s reason=bad_route_secret", _call_ref(call_id))
            return
        if active_calls >= settings.max_concurrent_calls:
            await websocket.accept()
            await websocket.send_text(config_frame)
            await websocket.send_text(
                dump_outbound(
                    ResponseOut(
                        response_id=0, content=BUSY_LINE, content_complete=True, end_call=True
                    )
                )
            )
            await websocket.close()
            logger.warning("ws rejected call=%s reason=busy", _call_ref(call_id))
            return
        active_calls += 1  # atomic with the capacity check above: no await between
        hermes: HermesClient = app.state.hermes
        sessions: set[CallSession] = app.state.sessions
        started = time.monotonic()
        session: CallSession | None = None
        outcome = "server_error"
        try:
            await websocket.accept()
            await websocket.send_text(config_frame)
            session = CallSession(
                call_id=call_id, settings=settings, hermes=hermes, send=websocket.send_text
            )
            sessions.add(session)
            malformed = 0
            while True:
                raw = await websocket.receive_text()
                try:
                    event = parse_inbound(raw, max_bytes=settings.max_ws_message_bytes)
                except RetellProtocolError as exc:
                    # Log the error type only; never frame content (may carry transcript).
                    malformed += 1
                    logger.warning(
                        "bad frame call=%s error=%s consecutive=%d",
                        _call_ref(call_id),
                        type(exc).__name__,
                        malformed,
                    )
                    if malformed >= MAX_CONSECUTIVE_MALFORMED:
                        outcome = "protocol_error"
                        await websocket.close(code=1008)
                        break
                    continue
                malformed = 0
                # Turn work runs in a task inside the session; dispatch stays fast so
                # ping_pong frames are always answered within Retell's window.
                await session.on_event(event)
        except WebSocketDisconnect:
            outcome = "disconnect"
        except Exception:
            logger.exception("ws loop failed call=%s", _call_ref(call_id))
        finally:
            if session is not None:
                sessions.discard(session)
                try:
                    await session.close()
                except asyncio.CancelledError:
                    # External cancellation (server shutdown, test-client teardown)
                    # while draining cleanup: the session's background reaping tasks
                    # keep running on the loop and still deliver the Hermes stop, so
                    # absorbing here never orphans a run.
                    logger.debug("session close interrupted call=%s", _call_ref(call_id))
                except Exception as exc:
                    logger.warning(
                        "session close failed call=%s error=%s",
                        _call_ref(call_id),
                        type(exc).__name__,
                    )
                if session.outcome == "denied":
                    outcome = "denied"
            active_calls -= 1
            duration_ms = int((time.monotonic() - started) * 1000)
            logger.info(
                "call end call=%s duration_ms=%d turns=%d outcome=%s",
                _call_ref(call_id),
                duration_ms,
                session.turns if session is not None else 0,
                outcome,
            )

    return app
