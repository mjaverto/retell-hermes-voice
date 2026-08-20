"""Integration tests for the Retell Custom LLM WebSocket route in server.py.

Drives the real ASGI app (``create_app``) through ``starlette.testclient.TestClient``,
with Hermes traffic routed into the programmable fake in ``tests/fake_hermes.py`` via
``build_fake_hermes_transport``, which streams SSE bodies incrementally (unlike
``httpx.ASGITransport``'s full-response buffering -- see that module's docstring).
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import re
import threading
import time
from collections.abc import Iterator
from queue import Empty, Queue
from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from starlette.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from fake_hermes import FakeHermesState, FakeScript, build_fake_hermes_transport
from retell_hermes_voice.call_session import DENIED_LINE
from retell_hermes_voice.config import Settings
from retell_hermes_voice.hermes_client import SAFE_TIMEOUT_MESSAGE
from retell_hermes_voice.server import BUSY_LINE, MAX_CONSECUTIVE_MALFORMED, create_app

# --- fixtures / helpers -----------------------------------------------------


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


def recv(ws: Any, timeout: float = 2.0) -> dict[str, Any]:
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


@contextlib.contextmanager
def open_call(client: TestClient, call_id: str, route_secret: str) -> Iterator[Any]:
    """Connect, consume + assert the mandatory first config frame, then yield the ws."""
    with client.websocket_connect(f"/llm-websocket/{route_secret}/{call_id}") as ws:
        config = recv(ws)
        assert config["response_type"] == "config"
        assert config["config"]["auto_reconnect"] is True
        assert config["config"]["call_details"] is True
        yield ws


def utterance(role: str, content: str) -> dict[str, str]:
    return {"role": role, "content": content}


def call_details_payload(
    *, call_id: str, from_number: str | None = None, direction: str = "inbound"
) -> dict[str, Any]:
    call: dict[str, Any] = {"call_id": call_id, "direction": direction, "call_type": "phone_call"}
    if from_number is not None:
        call["from_number"] = from_number
    return {"interaction_type": "call_details", "call": call}


def response_required_payload(
    response_id: int, *, user_content: str, history: list[dict[str, str]] | None = None
) -> dict[str, Any]:
    transcript = [*(history or []), utterance("user", user_content)]
    return {
        "interaction_type": "response_required",
        "response_id": response_id,
        "transcript": transcript,
    }


def ping_pong_payload(timestamp: int) -> dict[str, Any]:
    return {"interaction_type": "ping_pong", "timestamp": timestamp}


def collect_turn(ws: Any, response_id: int, deadline: float = 2.0) -> list[dict[str, Any]]:
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


# --- 1. normal flow: greeting then a full turn -----------------------------


def test_normal_flow_greeting_then_turn() -> None:
    script = FakeScript(deltas=["Paris", " is the capital."], delta_interval_s=0.0)
    with running_app(script) as (client, settings, _state):
        secret = settings.route_secret.get_secret_value()
        with open_call(client, "call-normal", secret) as ws:
            ws.send_json(call_details_payload(call_id="call-normal", from_number="+15551230000"))
            greeting = recv(ws)
            assert greeting["response_type"] == "response"
            assert greeting["response_id"] == 0
            assert greeting["content"] == settings.greeting
            assert greeting["content_complete"] is True

            ws.send_json(
                response_required_payload(1, user_content="What is the capital of France?")
            )
            frames = collect_turn(ws, response_id=1)
            combined = "".join(frame["content"] for frame in frames)
            assert "Paris" in combined
            assert frames[-1]["content_complete"] is True


# --- 2. ping/pong keepalive --------------------------------------------------


def test_ping_pong_echoes_same_timestamp() -> None:
    script = FakeScript(deltas=[], delta_interval_s=0.0)
    with running_app(script) as (client, settings, _state):
        secret = settings.route_secret.get_secret_value()
        with open_call(client, "call-ping", secret) as ws:
            timestamp = 1703302407333
            ws.send_json(ping_pong_payload(timestamp))
            frame = recv(ws)
            assert frame["response_type"] == "ping_pong"
            assert frame["timestamp"] == timestamp


# --- 3. wrong route secret ---------------------------------------------------


def test_wrong_route_secret_closes_1008_without_processing() -> None:
    script = FakeScript(deltas=[], delta_interval_s=0.0)
    with running_app(script) as (client, settings, _state):
        assert settings.route_secret.get_secret_value() != "WRONG-SECRET-VALUE-000"
        with client.websocket_connect("/llm-websocket/WRONG-SECRET-VALUE-000/call-bad") as ws:
            with pytest.raises(WebSocketDisconnect) as exc_info:
                recv(ws)
            assert exc_info.value.code == 1008


# --- 4. unauthorized caller: denied + turns ignored -------------------------


def test_unauthorized_caller_denied_and_turns_ignored() -> None:
    script = FakeScript(deltas=["should never be heard"], delta_interval_s=0.0)
    with running_app(script, allowed_callers=["+15551230000"]) as (client, settings, _state):
        secret = settings.route_secret.get_secret_value()
        with open_call(client, "call-denied", secret) as ws:
            ws.send_json(call_details_payload(call_id="call-denied", from_number="+15559999999"))
            denied = recv(ws)
            assert denied["response_type"] == "response"
            assert denied["end_call"] is True
            assert denied["content"] == DENIED_LINE
            assert "goodbye" in denied["content"].lower()

            # response_required is a silent no-op once denied; prove it with a
            # sentinel ping_pong instead of waiting on a frame that never comes.
            ws.send_json(response_required_payload(1, user_content="Are you there?"))
            timestamp = 4242
            ws.send_json(ping_pong_payload(timestamp))
            echo = recv(ws)
            assert echo["response_type"] == "ping_pong"
            assert echo["timestamp"] == timestamp


# --- 5. allowlist disabled: any caller greeted ------------------------------


def test_allowlist_disabled_greets_any_caller() -> None:
    script = FakeScript(deltas=[], delta_interval_s=0.0)
    with running_app(script, allowed_callers=[]) as (client, settings, _state):
        secret = settings.route_secret.get_secret_value()
        with open_call(client, "call-open", secret) as ws:
            ws.send_json(call_details_payload(call_id="call-open", from_number="+19998887777"))
            greeting = recv(ws)
            assert greeting["response_id"] == 0
            assert greeting["content"] == settings.greeting
            assert greeting["content_complete"] is True


# --- 6. interruption: a newer response_required supersedes and stops -------


def test_newer_turn_supersedes_stale_turn_and_stops_it() -> None:
    script = FakeScript(deltas=["one", "two", "three", "four", "five"], delta_interval_s=0.05)
    with running_app(script) as (client, settings, state):
        secret = settings.route_secret.get_secret_value()
        with open_call(client, "call-stale", secret) as ws:
            ws.send_json(call_details_payload(call_id="call-stale", from_number="+15551230000"))
            recv(ws)  # greeting

            ws.send_json(response_required_payload(1, user_content="Tell me a long story."))
            first = recv(ws, timeout=2.0)
            assert first["response_id"] == 1
            assert first["content_complete"] is False

            ws.send_json(response_required_payload(2, user_content="Actually, what time is it?"))
            collected: list[dict[str, Any]] = []
            deadline = time.monotonic() + 2.0
            saw_id2_complete = False
            while time.monotonic() < deadline:
                frame = recv(ws, timeout=max(0.05, deadline - time.monotonic()))
                collected.append(frame)
                if frame["response_id"] == 2 and frame["content_complete"]:
                    saw_id2_complete = True
                    break
            assert saw_id2_complete, "expected a content_complete frame for response_id=2"

            first_id2_index = next(
                index for index, frame in enumerate(collected) if frame["response_id"] == 2
            )
            assert all(frame["response_id"] == 2 for frame in collected[first_id2_index:]), (
                "no response_id=1 frame may arrive after the first response_id=2 frame"
            )

            stop_deadline = time.monotonic() + 2.0
            while time.monotonic() < stop_deadline and len(state.stops) < 1:
                time.sleep(0.02)
            assert len(state.stops) >= 1


# --- 7. tool call during an active turn: filler while waiting ---------------


def test_tool_call_emits_filler_before_completing() -> None:
    script = FakeScript(deltas=["Done."], delta_interval_s=0.2, tool_event_after=0)
    with running_app(script, filler_after_seconds=0.05) as (client, settings, _state):
        secret = settings.route_secret.get_secret_value()
        with open_call(client, "call-tool", secret) as ws:
            ws.send_json(call_details_payload(call_id="call-tool", from_number="+15551230000"))
            recv(ws)  # greeting

            ws.send_json(response_required_payload(1, user_content="Look something up."))
            frames = collect_turn(ws, response_id=1, deadline=3.0)

            filler_texts = {phrase.strip() for phrase in settings.filler_phrases}
            filler_present = any(
                not frame["content_complete"] and frame["content"].strip() in filler_texts
                for frame in frames
            )
            assert filler_present, "expected a filler frame while the tool call was in flight"

            combined = "".join(frame["content"] for frame in frames)
            assert "Done." in combined
            assert frames[-1]["content_complete"] is True


# --- 8. Hermes first-token timeout: safe apology, run stopped --------------


def test_hermes_timeout_yields_safe_message_and_stops_run() -> None:
    script = FakeScript(deltas=[], hang_after=0)
    with running_app(script, hermes_first_token_timeout=0.3) as (client, settings, state):
        secret = settings.route_secret.get_secret_value()
        with open_call(client, "call-timeout", secret) as ws:
            ws.send_json(call_details_payload(call_id="call-timeout", from_number="+15551230000"))
            recv(ws)  # greeting

            ws.send_json(response_required_payload(1, user_content="Anyone there?"))
            frames = collect_turn(ws, response_id=1, deadline=3.0)
            assert frames[-1]["content"] == SAFE_TIMEOUT_MESSAGE
            assert frames[-1]["content_complete"] is True

            stop_deadline = time.monotonic() + 2.0
            while time.monotonic() < stop_deadline and len(state.stops) < 1:
                time.sleep(0.02)
            assert len(state.stops) == 1
            assert state.stops[0]["run_id"] == state.runs[0]["run_id"]


# --- 9. client disconnect mid-turn: run stopped, no orphan ------------------


def test_client_disconnect_mid_turn_stops_run() -> None:
    script = FakeScript(deltas=["one", "two", "three", "four", "five", "six"], delta_interval_s=0.1)
    with running_app(script) as (client, settings, state):
        secret = settings.route_secret.get_secret_value()
        with client.websocket_connect(f"/llm-websocket/{secret}/call-drop") as ws:
            recv(ws)  # config
            ws.send_json(call_details_payload(call_id="call-drop", from_number="+15551230000"))
            recv(ws)  # greeting

            ws.send_json(response_required_payload(1, user_content="Keep talking."))
            first = recv(ws, timeout=2.0)
            assert first["response_id"] == 1
            # exiting the block below closes the socket from the client side

        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline and len(state.stops) < 1:
            time.sleep(0.02)
        assert len(state.stops) >= 1


# --- 10. malformed frames: threshold close, and counter reset --------------


def test_three_consecutive_malformed_frames_close_1008() -> None:
    script = FakeScript(deltas=[], delta_interval_s=0.0)
    with running_app(script) as (client, settings, _state):
        secret = settings.route_secret.get_secret_value()
        with open_call(client, "call-malformed", secret) as ws:
            for _ in range(MAX_CONSECUTIVE_MALFORMED):
                ws.send_text("not json")
            with pytest.raises(WebSocketDisconnect) as exc_info:
                recv(ws)
            assert exc_info.value.code == 1008


def test_malformed_counter_resets_on_a_valid_frame() -> None:
    script = FakeScript(deltas=[], delta_interval_s=0.0)
    with running_app(script) as (client, settings, _state):
        secret = settings.route_secret.get_secret_value()
        with open_call(client, "call-malformed-reset", secret) as ws:
            assert MAX_CONSECUTIVE_MALFORMED > 1
            for _ in range(MAX_CONSECUTIVE_MALFORMED - 1):
                ws.send_text("not json")
            timestamp = 111
            ws.send_json(ping_pong_payload(timestamp))
            echo = recv(ws)
            assert echo["response_type"] == "ping_pong"
            assert echo["timestamp"] == timestamp

            # the counter reset: another run of (threshold - 1) malformed frames
            # still doesn't close the connection.
            for _ in range(MAX_CONSECUTIVE_MALFORMED - 1):
                ws.send_text("not json")
            timestamp2 = 222
            ws.send_json(ping_pong_payload(timestamp2))
            echo2 = recv(ws)
            assert echo2["response_type"] == "ping_pong"
            assert echo2["timestamp"] == timestamp2


# --- 11. oversized payload: ignored, connection stays open ------------------


def test_oversized_payload_ignored_connection_stays_open() -> None:
    script = FakeScript(deltas=[], delta_interval_s=0.0)
    with running_app(script, max_ws_message_bytes=200) as (client, settings, _state):
        secret = settings.route_secret.get_secret_value()
        with open_call(client, "call-oversized", secret) as ws:
            padded = response_required_payload(1, user_content="x" * 500)
            raw = json.dumps(padded)
            assert len(raw.encode("utf-8")) > settings.max_ws_message_bytes
            ws.send_text(raw)

            timestamp = 99
            ws.send_json(ping_pong_payload(timestamp))
            echo = recv(ws)
            assert echo["response_type"] == "ping_pong"
            assert echo["timestamp"] == timestamp


# --- 12. concurrent calls: isolated Hermes sessions -------------------------


def test_concurrent_calls_have_isolated_sessions() -> None:
    script = FakeScript(deltas=["ok"], delta_interval_s=0.0)
    with running_app(script, max_concurrent_calls=2) as (client, settings, state):
        secret = settings.route_secret.get_secret_value()
        with (
            open_call(client, "call-iso-a", secret) as ws_a,
            open_call(client, "call-iso-b", secret) as ws_b,
        ):
            ws_a.send_json(call_details_payload(call_id="call-iso-a", from_number="+15551230000"))
            recv(ws_a)
            ws_b.send_json(call_details_payload(call_id="call-iso-b", from_number="+15551230001"))
            recv(ws_b)

            ws_a.send_json(response_required_payload(1, user_content="Hi from A"))
            collect_turn(ws_a, response_id=1)
            ws_b.send_json(response_required_payload(1, user_content="Hi from B"))
            collect_turn(ws_b, response_id=1)

        assert len(state.runs) == 2
        headers_a, headers_b = state.runs[0]["headers"], state.runs[1]["headers"]
        session_id_a = headers_a["x-hermes-session-id"]
        session_id_b = headers_b["x-hermes-session-id"]
        session_key_a = headers_a["x-hermes-session-key"]
        session_key_b = headers_b["x-hermes-session-key"]
        assert session_id_a != session_id_b
        assert session_key_a != session_key_b

        body_session_a = state.runs[0]["body"]["session_id"]
        body_session_b = state.runs[1]["body"]["session_id"]
        assert body_session_a != body_session_b
        assert body_session_a == session_id_a
        assert body_session_b == session_id_b

        raw_identifiers = {"call-iso-a", "call-iso-b", "+15551230000", "+15551230001"}
        for value in (session_id_a, session_id_b, session_key_a, session_key_b):
            assert value not in raw_identifiers


# --- 13. busy line: over-capacity call rejected -----------------------------


def test_busy_line_rejects_over_capacity_call() -> None:
    script = FakeScript(deltas=["hi"], delta_interval_s=0.0)
    with running_app(script, max_concurrent_calls=1) as (client, settings, _state):
        secret = settings.route_secret.get_secret_value()
        with (
            open_call(client, "call-holds-slot", secret),
            client.websocket_connect(f"/llm-websocket/{secret}/call-busy") as ws2,
        ):
            config = recv(ws2)
            assert config["response_type"] == "config"
            busy = recv(ws2)
            assert busy["response_type"] == "response"
            assert busy["end_call"] is True
            assert busy["content"] == BUSY_LINE
            with pytest.raises(WebSocketDisconnect):
                recv(ws2)


# --- 14. healthz / readyz ----------------------------------------------------


def test_healthz_always_ok() -> None:
    script = FakeScript(deltas=[], delta_interval_s=0.0)
    with running_app(script) as (client, _settings, _state):
        response = client.get("/healthz")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


def test_readyz_200_when_hermes_healthy() -> None:
    script = FakeScript(deltas=[], delta_interval_s=0.0)
    with running_app(script) as (client, _settings, _state):
        response = client.get("/readyz")
        assert response.status_code == 200
        assert response.json()["status"] == "ready"


def test_readyz_503_when_hermes_unhealthy() -> None:
    def build_unhealthy_hermes() -> FastAPI:
        app = FastAPI()

        @app.get("/health")
        async def health() -> JSONResponse:
            return JSONResponse({"status": "error"}, status_code=500)

        return app

    settings = make_settings()
    app = create_app(settings, hermes_transport=httpx.ASGITransport(app=build_unhealthy_hermes()))
    with TestClient(app) as client:
        response = client.get("/readyz")
        assert response.status_code == 503


# --- 15. barge-in: superseded turn teardown never blocks the WS loop --------


def test_barge_in_teardown_does_not_block_ws_loop() -> None:
    """A superseded turn's slow Hermes stop must not delay ping_pong echoes.

    Regression: cancelling the old turn used to be awaited inline, so a slow
    POST /stop (bounded by hermes_stop_timeout) starved the receive loop past
    Retell's 5 s ping deadline.
    """
    script = FakeScript(deltas=["one", "two", "three"], delta_interval_s=0.2, stop_delay_s=0.5)
    with running_app(script) as (client, settings, state):
        secret = settings.route_secret.get_secret_value()
        with open_call(client, "call-barge", secret) as ws:
            ws.send_json(call_details_payload(call_id="call-barge", from_number="+15551230000"))
            recv(ws)  # greeting

            ws.send_json(response_required_payload(1, user_content="Tell me a long story."))
            first = recv(ws, timeout=2.0)
            assert first["response_id"] == 1

            ws.send_json(response_required_payload(2, user_content="Actually, stop."))
            timestamp = 555
            ws.send_json(ping_pong_payload(timestamp))

            # The echo must arrive before turn 2 completes and while turn 1's
            # delayed stop is still pending (i.e. teardown was not awaited).
            saw_echo = False
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline:
                frame = recv(ws, timeout=max(0.05, deadline - time.monotonic()))
                if frame.get("response_type") == "ping_pong":
                    saw_echo = True
                    assert frame["timestamp"] == timestamp
                    break
                assert not (frame["response_id"] == 2 and frame["content_complete"]), (
                    "ping_pong echo must not wait for the new turn to finish"
                )
            assert saw_echo
            assert len(state.stops) == 0, (
                "turn 1's Hermes stop must still be in flight when the echo arrives"
            )

            collect_turn(ws, response_id=2, deadline=3.0)

            # Teardown still completes in the background: the stop lands.
            stop_deadline = time.monotonic() + 3.0
            while time.monotonic() < stop_deadline and len(state.stops) < 1:
                time.sleep(0.02)
            assert len(state.stops) >= 1


# --- 16. enforced allowlist: turn before call_details is denied --------------


def test_enforced_allowlist_denies_turn_before_call_details() -> None:
    script = FakeScript(deltas=["must never run"], delta_interval_s=0.0)
    with running_app(script, allowed_callers=["+15551230000"]) as (client, settings, state):
        secret = settings.route_secret.get_secret_value()
        with open_call(client, "call-early-turn", secret) as ws:
            # response_required arrives before call_details: the allowlist cannot
            # be checked yet, so the call is denied instead of running a turn.
            ws.send_json(response_required_payload(1, user_content="Hello?"))
            denied = recv(ws)
            assert denied["response_type"] == "response"
            assert denied["response_id"] == 1
            assert denied["content"] == DENIED_LINE
            assert denied["content_complete"] is True
            assert denied["end_call"] is True
        assert state.runs == []


# --- 17. reminder_required: synthetic nudge instead of stale input -----------


def test_reminder_required_runs_turn_with_synthetic_nudge() -> None:
    script = FakeScript(deltas=["Are you still there?"], delta_interval_s=0.0)
    with running_app(script) as (client, settings, state):
        secret = settings.route_secret.get_secret_value()
        with open_call(client, "call-reminder", secret) as ws:
            ws.send_json(call_details_payload(call_id="call-reminder", from_number="+15551230000"))
            recv(ws)  # greeting

            ws.send_json(
                {
                    "interaction_type": "reminder_required",
                    "response_id": 1,
                    "transcript": [utterance("agent", "How can I help you today?")],
                }
            )
            frames = collect_turn(ws, response_id=1)
            assert frames[-1]["content_complete"] is True

        assert len(state.runs) == 1
        assert "gone quiet" in state.runs[-1]["body"]["input"]


# --- 18. ttfb log: sane, single-clock value ----------------------------------


def test_ttfb_log_reports_sane_monotonic_value(caplog: pytest.LogCaptureFixture) -> None:
    script = FakeScript(deltas=["hi there"], delta_interval_s=0.0)
    with (
        caplog.at_level(logging.INFO, logger="retell_hermes_voice.call_session"),
        running_app(script) as (client, settings, _state),
    ):
        secret = settings.route_secret.get_secret_value()
        with open_call(client, "call-ttfb", secret) as ws:
            ws.send_json(call_details_payload(call_id="call-ttfb", from_number="+15551230000"))
            recv(ws)  # greeting
            ws.send_json(response_required_payload(1, user_content="Quick one."))
            collect_turn(ws, response_id=1)
    first_delta_messages = [
        record.getMessage() for record in caplog.records if "first_delta" in record.getMessage()
    ]
    assert first_delta_messages
    match = re.search(r"ttfb_ms=(-?\d+)", first_delta_messages[0])
    assert match is not None
    ttfb_ms = int(match.group(1))
    # Mixing loop.time() with time.monotonic() yields garbage (huge or negative)
    # under uvloop; an in-process turn against the fake stays well under 5 s.
    assert 0 <= ttfb_ms < 5000


# --- 19. denied-call logs: hashed ref only, never the raw call id ------------


def test_denied_call_logs_hash_ref_never_raw_call_id(caplog: pytest.LogCaptureFixture) -> None:
    script = FakeScript(deltas=[], delta_interval_s=0.0)
    call_id = "call-raw-secret-xyz-987"
    expected_ref = hashlib.sha256(call_id.encode()).hexdigest()[:12]
    with (
        caplog.at_level(logging.INFO),
        running_app(script, allowed_callers=["+15551230000"]) as (client, settings, _state),
    ):
        secret = settings.route_secret.get_secret_value()
        with open_call(client, call_id, secret) as ws:
            ws.send_json(call_details_payload(call_id=call_id, from_number="+15559999999"))
            denied = recv(ws)
            assert denied["content"] == DENIED_LINE
    messages = [record.getMessage() for record in caplog.records]
    assert any(expected_ref in message for message in messages)
    assert all(call_id not in message for message in messages)
