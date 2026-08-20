"""End-to-end smoke test for the Retell <-> Hermes voice adapter.

Drives one realistic call -- connect, config, call_details, greeting, a
sanitized multi-delta answer, a mid-call keepalive, a second turn, and a clean
hangup -- through the real ASGI stack (``create_app``) against the
programmable fake Hermes in ``tests/fake_hermes.py``. Complements the
scenario-by-scenario coverage in ``tests/test_server_ws.py``.
"""

from __future__ import annotations

import contextlib
import threading
import time
from collections.abc import Iterator
from queue import Empty, Queue
from typing import Any

from starlette.testclient import TestClient

from fake_hermes import FakeHermesState, FakeScript, build_fake_hermes_transport
from retell_hermes_voice.config import Settings
from retell_hermes_voice.server import create_app

_ALLOWED_NUMBER = "+15551230000"


def make_settings(**overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "hermes_base_url": "http://fake-hermes.invalid",
        "hermes_api_key": "test-api-key",
        "route_secret": "route-secret-0123456789",
        "warmup_on_start": False,
        "hermes_first_token_timeout": 2.0,
        "hermes_turn_timeout": 5.0,
        "hermes_stop_timeout": 1.0,
        "filler_after_seconds": 5.0,
        "allowed_callers": [_ALLOWED_NUMBER],
        "_env_file": None,
    }
    values.update(overrides)
    return Settings(**values)


@contextlib.contextmanager
def running_app(
    script: FakeScript, **settings_overrides: Any
) -> Iterator[tuple[TestClient, Settings, FakeHermesState]]:
    """Build a real app wired to a fake Hermes and run its lifespan for the block."""
    transport, state = build_fake_hermes_transport(script)
    settings = make_settings(**settings_overrides)
    app = create_app(settings, hermes_transport=transport)
    with TestClient(app) as client:
        yield client, settings, state


def recv(ws: Any, timeout: float = 3.0) -> dict[str, Any]:
    """Read one JSON frame from a WebSocketTestSession, bounded by ``timeout``.

    ``WebSocketTestSession.receive_json`` blocks indefinitely with no timeout hook,
    and its internal transport has changed across Starlette versions (queue-based vs.
    anyio streams). Running the public, version-stable ``receive_json()`` call on a
    daemon thread and waiting on a plain queue bounds the wait without depending on
    any private implementation detail; a server bug that never sends the expected
    frame fails the test instead of hanging the whole suite.
    """
    outcome: Queue[tuple[str, Any]] = Queue(maxsize=1)

    def worker() -> None:
        try:
            outcome.put(("ok", ws.receive_json()))
        except BaseException as exc:  # forwarded to the waiting thread below
            outcome.put(("error", exc))

    threading.Thread(target=worker, daemon=True).start()
    try:
        status, payload = outcome.get(timeout=timeout)
    except Empty:
        raise AssertionError(f"no websocket frame received within {timeout}s") from None
    if status == "error":
        raise payload  # type: ignore[misc]
    return payload  # type: ignore[no-any-return]


def utterance(role: str, content: str) -> dict[str, str]:
    return {"role": role, "content": content}


def call_details_payload(*, call_id: str, from_number: str) -> dict[str, Any]:
    return {
        "interaction_type": "call_details",
        "call": {
            "call_id": call_id,
            "direction": "inbound",
            "call_type": "phone_call",
            "from_number": from_number,
        },
    }


def response_required_payload(
    response_id: int, *, user_content: str, history: list[dict[str, str]]
) -> dict[str, Any]:
    transcript = [*history, utterance("user", user_content)]
    return {
        "interaction_type": "response_required",
        "response_id": response_id,
        "transcript": transcript,
    }


def ping_pong_payload(timestamp: int) -> dict[str, Any]:
    return {"interaction_type": "ping_pong", "timestamp": timestamp}


def collect_turn(ws: Any, response_id: int, deadline: float = 3.0) -> list[dict[str, Any]]:
    """Read response frames for ``response_id`` until content_complete, or fail."""
    frames: list[dict[str, Any]] = []
    start = time.monotonic()
    while True:
        remaining = deadline - (time.monotonic() - start)
        if remaining <= 0:
            raise AssertionError(f"timed out collecting turn frames for response_id={response_id}")
        frame = recv(ws, timeout=remaining)
        assert frame["response_type"] == "response"
        assert frame["response_id"] == response_id
        frames.append(frame)
        if frame["content_complete"]:
            return frames


def test_full_call_end_to_end_through_real_asgi_stack() -> None:
    """One realistic call against the real stack: greeting, a sanitized multi-delta
    answer, a mid-call keepalive, a second turn, and a hangup with no orphaned runs."""
    # Markdown constructs are kept whole within a delta and never end a chunk on a
    # bare backtick/star, matching how DeltaSanitizer resolves streamed text.
    script = FakeScript(
        deltas=["The capital is ", "**Paris**", ", built on the `Seine`."],
        delta_interval_s=0.02,
    )
    call_id = "call-smoke-1"
    with running_app(script) as (client, settings, state):
        secret = settings.route_secret.get_secret_value()
        with client.websocket_connect(f"/llm-websocket/{secret}/{call_id}") as ws:
            config = recv(ws)
            assert config["response_type"] == "config"
            assert config["config"]["auto_reconnect"] is True
            assert config["config"]["call_details"] is True

            ws.send_json(call_details_payload(call_id=call_id, from_number=_ALLOWED_NUMBER))
            greeting = recv(ws)
            assert greeting["response_type"] == "response"
            assert greeting["response_id"] == 0
            assert greeting["content"] == settings.greeting
            assert greeting["content_complete"] is True

            history = [utterance("agent", settings.greeting)]
            ws.send_json(
                response_required_payload(
                    1, user_content="What's the capital of France?", history=history
                )
            )
            turn1_frames = collect_turn(ws, response_id=1)
            turn1_speech = "".join(frame["content"] for frame in turn1_frames)
            assert "*" not in turn1_speech
            assert "`" not in turn1_speech
            assert "Paris" in turn1_speech
            assert "Seine" in turn1_speech

            ping_timestamp = 1700000000123
            ws.send_json(ping_pong_payload(ping_timestamp))
            echo = recv(ws)
            assert echo["response_type"] == "ping_pong"
            assert echo["timestamp"] == ping_timestamp

            history.extend(
                [
                    utterance("user", "What's the capital of France?"),
                    utterance("agent", turn1_speech),
                ]
            )
            ws.send_json(
                response_required_payload(2, user_content="Thanks, that's all.", history=history)
            )
            turn2_frames = collect_turn(ws, response_id=2)
            assert turn2_frames[-1]["content_complete"] is True
            # hangup: exiting this block closes the socket from the client side.

        # every observed frame is one of the three well-formed outbound kinds
        all_frames = [config, greeting, *turn1_frames, echo, *turn2_frames]
        for frame in all_frames:
            assert frame["response_type"] in {"config", "response", "ping_pong"}

        # response ids echoed correctly: greeting=0, turn1=1, turn2=2
        assert greeting["response_id"] == 0
        assert {frame["response_id"] for frame in turn1_frames} == {1}
        assert {frame["response_id"] for frame in turn2_frames} == {2}

        # exactly one Hermes run per turn, both sharing the same stable per-call session
        assert len(state.runs) == 2
        session_ids = {run["headers"]["x-hermes-session-id"] for run in state.runs}
        session_keys = {run["headers"]["x-hermes-session-key"] for run in state.runs}
        body_session_ids = {run["body"]["session_id"] for run in state.runs}
        assert len(session_ids) == 1
        assert len(session_keys) == 1
        assert body_session_ids == session_ids

        # both turns completed cleanly -- no stop was needed for either
        assert state.stops == []

        # after hangup: no stop storm in the quiet period that follows
        stops_after_close = len(state.stops)
        time.sleep(0.5)
        assert len(state.stops) == stops_after_close
