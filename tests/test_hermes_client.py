"""Unit tests for retell_hermes_voice.hermes_client.

All Hermes traffic goes through httpx.ASGITransport into the programmable fake in
tests/fake_hermes.py -- no network, no real endpoints. Note that ASGITransport runs
the ASGI app to completion inside each request, so hang scripts exercise the client's
asyncio deadlines around the SSE subscribe itself.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from fake_hermes import FakeHermesState, FakeScript, build_fake_hermes
from retell_hermes_voice.config import Settings
from retell_hermes_voice.hermes_client import (
    HermesClient,
    HermesTurnEvent,
    HermesUnavailableError,
)

TIMEOUT_TEXT = "Sorry, that's taking longer than expected. Could you say that again?"
BUSY_TEXT = "I'm handling too many calls right now. Please try again in a moment."


def make_settings(**overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "hermes_base_url": "http://fake-hermes.invalid",
        "hermes_api_key": "test-api-key",
        "route_secret": "route-secret-0123456789",
        "voice_model": "google/gemini-3.7-flash",
        "voice_provider": "openrouter",
        "voice_reasoning_effort": "low",
        "hermes_first_token_timeout": 5.0,
        "hermes_turn_timeout": 10.0,
        "hermes_stop_timeout": 1.0,
        "warmup_on_start": False,
    }
    values.update(overrides)
    return Settings(**values)


def make_client(
    script: FakeScript, **settings_overrides: Any
) -> tuple[HermesClient, FakeHermesState]:
    app, state = build_fake_hermes(script)
    settings = make_settings(**settings_overrides)
    client = HermesClient(settings, transport=httpx.ASGITransport(app=app))
    return client, state


def turn_args(**overrides: Any) -> dict[str, Any]:
    args: dict[str, Any] = {
        "session_id": "rhv-test-session",
        "session_key": "rhv-test-key",
        "instructions": "Be terse.",
        "conversation_history": [{"role": "user", "content": "Hi."}],
        "user_input": "What is the capital of France?",
    }
    args.update(overrides)
    return args


async def drain(client: HermesClient) -> list[HermesTurnEvent]:
    return [event async for event in client.run_turn(**turn_args())]


async def test_happy_path_yields_deltas_then_done() -> None:
    client, state = make_client(FakeScript(deltas=["Hel", "lo", "!"], delta_interval_s=0.0))
    try:
        events = await drain(client)
    finally:
        await client.aclose()
    assert [event.kind for event in events] == ["delta", "delta", "delta", "done"]
    assert "".join(event.text for event in events if event.kind == "delta") == "Hello!"
    assert events[-1].text == "Hello!"


async def test_done_path_does_not_call_stop() -> None:
    client, state = make_client(FakeScript(deltas=["fin"], delta_interval_s=0.0))
    try:
        events = await drain(client)
    finally:
        await client.aclose()
    assert events[-1].kind == "done"
    assert state.stops == []


async def test_create_request_carries_routing_and_session_headers() -> None:
    client, state = make_client(FakeScript(deltas=["ok"], delta_interval_s=0.0))
    try:
        await drain(client)
    finally:
        await client.aclose()
    run = state.runs[0]
    body = run["body"]
    assert body["input"] == "What is the capital of France?"
    assert body["session_id"] == "rhv-test-session"
    assert body["instructions"] == "Be terse."
    assert body["conversation_history"] == [{"role": "user", "content": "Hi."}]
    assert body["model"] == "google/gemini-3.7-flash"
    assert body["provider"] == "openrouter"
    assert body["model_options"] == {"reasoning_effort": "low"}
    headers = run["headers"]
    assert headers["x-hermes-session-id"] == "rhv-test-session"
    assert headers["x-hermes-session-key"] == "rhv-test-key"
    assert headers["authorization"] == "Bearer test-api-key"


async def test_create_request_omits_routing_when_unset() -> None:
    client, state = make_client(
        FakeScript(deltas=["ok"], delta_interval_s=0.0),
        voice_model=None,
        voice_provider=None,
        voice_reasoning_effort=None,
    )
    try:
        await drain(client)
    finally:
        await client.aclose()
    body = state.runs[0]["body"]
    assert "model" not in body
    assert "provider" not in body
    assert "model_options" not in body


async def test_tool_started_surfaces_as_tool_start() -> None:
    client, state = make_client(
        FakeScript(deltas=["Hi", " there"], delta_interval_s=0.0, tool_event_after=1)
    )
    try:
        events = await drain(client)
    finally:
        await client.aclose()
    assert [event.kind for event in events] == ["delta", "tool_start", "delta", "done"]
    tool_event = next(event for event in events if event.kind == "tool_start")
    assert tool_event.text == ""


async def test_first_token_timeout_yields_error_and_stops_run() -> None:
    client, state = make_client(
        FakeScript(deltas=[], delta_interval_s=0.0, hang_after=0),
        hermes_first_token_timeout=0.2,
        hermes_turn_timeout=5.0,
    )
    try:
        events = await drain(client)
    finally:
        await client.aclose()
    assert [event.kind for event in events] == ["error"]
    assert events[0].text == TIMEOUT_TEXT
    assert len(state.stops) == 1
    assert state.stops[0]["run_id"] == state.runs[0]["run_id"]


async def test_turn_timeout_yields_error_and_stops_run() -> None:
    # The stream hangs after the first delta; the total turn deadline must fire.
    # (ASGITransport buffers responses, so the delta may never reach the client;
    # only the terminal error and the stop call are transport-independent.)
    client, state = make_client(
        FakeScript(deltas=["a", "b"], delta_interval_s=0.0, hang_after=1),
        hermes_first_token_timeout=5.0,
        hermes_turn_timeout=0.2,
    )
    try:
        events = await drain(client)
    finally:
        await client.aclose()
    assert events[-1].kind == "error"
    assert events[-1].text == TIMEOUT_TEXT
    assert all(event.kind == "delta" for event in events[:-1])
    assert len(state.stops) == 1


async def test_consumer_abort_stops_run_exactly_once() -> None:
    client, state = make_client(FakeScript(deltas=["one", "two", "three"], delta_interval_s=0.0))
    try:
        agen = client.run_turn(**turn_args())
        first: HermesTurnEvent | None = None
        async for event in agen:
            if event.kind == "delta":
                first = event
                break
        assert first is not None
        assert first.text == "one"
        await agen.aclose()
        assert len(state.stops) == 1
        assert state.stops[0]["run_id"] == state.runs[0]["run_id"]
    finally:
        await client.aclose()


async def test_error_content_intercepted() -> None:
    raw = "Provider authentication failed: Unknown provider 'not-a-provider'. Check config."
    client, state = make_client(FakeScript(deltas=[], error_content=raw))
    try:
        events = await drain(client)
    finally:
        await client.aclose()
    assert [event.kind for event in events] == ["error"]
    for event in events:
        assert raw not in event.text
        assert "Provider authentication failed" not in event.text


async def test_error_content_with_warning_emoji_intercepted() -> None:
    raw = "\u26a0\ufe0f Provider authentication failed: Unknown provider 'nope'."
    client, state = make_client(FakeScript(deltas=[], error_content=raw))
    try:
        events = await drain(client)
    finally:
        await client.aclose()
    assert [event.kind for event in events] == ["error"]
    for event in events:
        assert "\u26a0" not in event.text
        assert "Provider authentication failed" not in event.text


async def test_429_create_yields_busy_error() -> None:
    client, state = make_client(FakeScript(deltas=[], fail_create=429))
    try:
        events = await drain(client)
    finally:
        await client.aclose()
    assert [event.kind for event in events] == ["error"]
    assert events[0].text == BUSY_TEXT
    assert state.runs == []
    assert state.stops == []


async def test_health_true_on_ok() -> None:
    client, state = make_client(FakeScript(deltas=[]))
    try:
        assert await client.health() is True
    finally:
        await client.aclose()
    assert state.health_calls == 1


async def test_health_raises_when_unreachable() -> None:
    def raise_connect(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    client = HermesClient(make_settings(), transport=httpx.MockTransport(raise_connect))
    try:
        with pytest.raises(HermesUnavailableError):
            await client.health()
    finally:
        await client.aclose()


async def test_warmup_drains_one_run_with_voice_routing() -> None:
    client, state = make_client(FakeScript(deltas=["OK"], delta_interval_s=0.0))
    try:
        await client.warmup()
    finally:
        await client.aclose()
    assert len(state.runs) == 1
    body = state.runs[0]["body"]
    assert body["input"] == "Say OK."
    assert body["instructions"] == "Reply with exactly: OK"
    assert body["model"] == "google/gemini-3.7-flash"
    assert body["provider"] == "openrouter"
    assert state.stops == []  # completed run is never stopped


async def test_warmup_swallows_errors() -> None:
    client, state = make_client(FakeScript(deltas=[], fail_create=429))
    try:
        await client.warmup()  # must not raise
    finally:
        await client.aclose()
    assert state.runs == []
