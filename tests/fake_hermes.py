"""Programmable fake Hermes API server for tests.

Implements the routes HermesClient uses -- ``POST /v1/runs``, ``GET
/v1/runs/{run_id}/events`` (SSE), ``POST /v1/runs/{run_id}/stop``, ``GET /health`` --
emitting the same wire shapes documented in hermes-contract.md (sections 1 and 9):
bare ``data:`` SSE frames whose JSON carries the event name in the ``event`` field
(``message.delta``, ``tool.started``, ``run.completed``), a 202 create response with a
``run_id``, keepalive/terminator comment lines, and the observed 429 error envelope.

Two ways to mount this fake:

- ``build_fake_hermes`` returns the raw FastAPI app for ``httpx.ASGITransport``.
  Simple, but ASGITransport awaits the whole ASGI call before returning a Response,
  so a StreamingResponse's chunks (and the ``asyncio.sleep`` pacing between them) are
  fully collapsed into that single await -- the caller never observes a run as "in
  progress", only ever fully done (or, for a hanging script, never done at all).
- ``build_fake_hermes_transport`` wraps the same app but streams the SSE events
  endpoint incrementally, restoring the shape of a real socket. Required by any test
  that asserts a turn is genuinely mid-flight when it is cancelled or the client
  disconnects (barge-in, supersede, disconnect-mid-turn): under ASGITransport those
  races are not reproducible -- the "turn" is always already complete by the time the
  caller can react, deterministically starving stop-delivery assertions.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse


@dataclass
class FakeScript:
    """Programmable behavior for one fake Hermes run."""

    deltas: list[str]
    delta_interval_s: float = 0.01
    tool_event_after: int | None = None  # emit tool.started after N deltas
    hang_after: int | None = None  # stop emitting after N deltas, keep the stream open
    error_content: str | None = None  # emit as a single delta (fail-open error body)
    fail_create: int | None = None  # HTTP status to return from POST /v1/runs
    completed_usage: dict[str, Any] | None = None  # overrides usage on run.completed
    pre_frames: list[str] = field(default_factory=list)  # raw SSE text emitted first
    fail_stop: int | None = None  # HTTP status to return from POST /v1/runs/{id}/stop
    hang_stop: bool = False  # never answer POST /v1/runs/{id}/stop
    stop_delay_s: float = 0.0  # delay before recording/answering a stop


@dataclass
class FakeHermesState:
    """Everything the fake observed, for assertions."""

    runs: list[dict[str, Any]] = field(default_factory=list)  # run_id, body, headers, time
    stops: list[dict[str, Any]] = field(default_factory=list)  # run_id, time
    health_calls: int = 0


def _frame(payload: dict[str, Any]) -> str:
    """One bare data: SSE frame, exactly like the live /v1/runs stream."""
    return f"data: {json.dumps(payload)}\n\n"


async def _hang_forever() -> None:
    # Cancellable "keep the stream open" -- the event is never set; the await
    # ends only when the client abandons the request (cancellation).
    await asyncio.Event().wait()


async def _events(script: FakeScript, run_id: str) -> AsyncIterator[str]:
    for pre_frame in script.pre_frames:
        yield pre_frame
    if script.error_content is not None:
        yield _frame(
            {
                "event": "message.delta",
                "run_id": run_id,
                "timestamp": time.time(),
                "delta": script.error_content,
            }
        )
        yield _frame(
            {
                "event": "run.completed",
                "run_id": run_id,
                "timestamp": time.time(),
                "output": script.error_content,
                "usage": script.completed_usage
                if script.completed_usage is not None
                else {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            }
        )
        yield ": stream closed\n\n"
        return
    emitted = ""
    for index, delta in enumerate(script.deltas):
        if script.hang_after is not None and index >= script.hang_after:
            await _hang_forever()
            return
        if script.tool_event_after is not None and index == script.tool_event_after:
            yield _frame(
                {
                    "event": "tool.started",
                    "run_id": run_id,
                    "timestamp": time.time(),
                    "tool": "terminal",
                    "preview": "fake command",
                }
            )
        if script.delta_interval_s > 0:
            await asyncio.sleep(script.delta_interval_s)
        yield _frame(
            {
                "event": "message.delta",
                "run_id": run_id,
                "timestamp": time.time(),
                "delta": delta,
            }
        )
        emitted += delta
    if script.hang_after is not None and script.hang_after >= len(script.deltas):
        await _hang_forever()
        return
    yield _frame(
        {
            "event": "run.completed",
            "run_id": run_id,
            "timestamp": time.time(),
            "output": emitted,
            "usage": script.completed_usage
            if script.completed_usage is not None
            else {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        }
    )
    yield ": stream closed\n\n"


def build_fake_hermes(script: FakeScript) -> tuple[FastAPI, FakeHermesState]:
    state = FakeHermesState()
    app = FastAPI()

    @app.get("/health")
    async def health() -> JSONResponse:
        state.health_calls += 1
        return JSONResponse({"status": "ok", "platform": "hermes-agent", "version": "0.20.4"})

    @app.post("/v1/runs")
    async def create_run(request: Request) -> JSONResponse:
        if script.fail_create is not None:
            return JSONResponse(
                {
                    "error": {
                        "message": "Too many concurrent runs (max 10)",
                        "type": "rate_limit_error",
                        "param": None,
                        "code": "rate_limit_exceeded",
                    }
                },
                status_code=script.fail_create,
                headers={"Retry-After": "1"},
            )
        body = await request.json()
        run_id = f"run_{uuid.uuid4().hex}"
        state.runs.append(
            {
                "run_id": run_id,
                "body": body,
                "headers": dict(request.headers),
                "time": time.monotonic(),
            }
        )
        return JSONResponse({"run_id": run_id, "status": "started"}, status_code=202)

    @app.get("/v1/runs/{run_id}/events")
    async def run_events(run_id: str) -> StreamingResponse:
        return StreamingResponse(
            _events(script, run_id),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.post("/v1/runs/{run_id}/stop")
    async def stop_run(run_id: str) -> JSONResponse:
        if script.stop_delay_s > 0:
            await asyncio.sleep(script.stop_delay_s)
        if script.hang_stop:
            await asyncio.Event().wait()
        state.stops.append({"run_id": run_id, "time": time.monotonic()})
        if script.fail_stop is not None:
            return JSONResponse(
                {"error": {"message": "stop rejected", "type": "test_error"}},
                status_code=script.fail_stop,
            )
        return JSONResponse({"run_id": run_id, "status": "stopping"})

    return app, state


class _SSEByteStream(httpx.AsyncByteStream):
    """Adapts an ``AsyncIterator[str]`` of SSE text into httpx's byte-stream protocol.

    Each frame is only encoded (and, for ``_events``, only produced -- the async
    generator does not run ahead of its consumer) when the caller actually pulls the
    next chunk, which is what makes this genuinely incremental.
    """

    def __init__(self, frames: AsyncIterator[str]) -> None:
        self._frames = frames

    async def __aiter__(self) -> AsyncIterator[bytes]:
        async for frame in self._frames:
            yield frame.encode("utf-8")


class _StreamingHermesTransport(httpx.AsyncBaseTransport):
    """``build_fake_hermes``'s app, except GET .../events streams incrementally.

    Request/response routes (create, stop, health) have no pacing to preserve, so
    they still go through ``httpx.ASGITransport`` unchanged; only the SSE endpoint
    needs a response that returns before its body is fully produced.
    """

    _EVENTS_PREFIX = "/v1/runs/"
    _EVENTS_SUFFIX = "/events"

    def __init__(self, app: FastAPI, script: FakeScript) -> None:
        self._inner = httpx.ASGITransport(app=app)
        self._script = script

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if (
            request.method == "GET"
            and path.startswith(self._EVENTS_PREFIX)
            and path.endswith(self._EVENTS_SUFFIX)
        ):
            run_id = path[len(self._EVENTS_PREFIX) : -len(self._EVENTS_SUFFIX)]
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream", "cache-control": "no-cache"},
                stream=_SSEByteStream(_events(self._script, run_id)),
            )
        return await self._inner.handle_async_request(request)

    async def aclose(self) -> None:
        await self._inner.aclose()


def build_fake_hermes_transport(
    script: FakeScript,
) -> tuple[httpx.AsyncBaseTransport, FakeHermesState]:
    """Like ``build_fake_hermes``, but the SSE events endpoint streams incrementally.

    Use this for any test that needs a turn to genuinely be mid-flight (still
    receiving deltas, not yet at ``run.completed``) at some observable moment --
    ``build_fake_hermes``'s ``httpx.ASGITransport`` cannot do that (see module
    docstring).
    """
    app, state = build_fake_hermes(script)
    return _StreamingHermesTransport(app, script), state
