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
    """Build the ASGI app; one HermesClient and one call semaphore per app.

    ``hermes_transport`` lets tests mount a fake Hermes backend in-process.
    """
    call_slots = asyncio.Semaphore(settings.max_concurrent_calls)
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
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await warmup_task
            sessions: set[CallSession] = app.state.sessions
            for session in list(sessions):
                with contextlib.suppress(Exception):
                    await session.close()
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
        except HermesUnavailableError:
            ready = False
        if ready:
            return JSONResponse({"status": "ready"})
        return JSONResponse({"status": "degraded"}, status_code=503)

    @app.websocket("/llm-websocket/{route_secret}/{call_id}")
    async def llm_websocket(websocket: WebSocket, route_secret: str, call_id: str) -> None:
        if not secrets.compare_digest(
            route_secret.encode(), settings.route_secret.get_secret_value().encode()
        ):
            # Starlette requires accept before close; no events are processed.
            await websocket.accept()
            await websocket.close(code=1008)
            logger.warning("ws rejected call=%s reason=bad_route_secret", _call_ref(call_id))
            return
        if call_slots.locked():
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
        await call_slots.acquire()
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
                with contextlib.suppress(Exception):
                    await session.close()
                if session.outcome == "denied":
                    outcome = "denied"
            call_slots.release()
            duration_ms = int((time.monotonic() - started) * 1000)
            logger.info(
                "call end call=%s duration_ms=%d turns=%d outcome=%s",
                _call_ref(call_id),
                duration_ms,
                session.turns if session is not None else 0,
                outcome,
            )

    return app
