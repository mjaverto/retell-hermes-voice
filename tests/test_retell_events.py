"""Unit tests for retell_hermes_voice.retell_events (Retell wire types)."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from retell_hermes_voice.retell_events import (
    CallDetailsEvent,
    ConfigOut,
    OversizedPayloadError,
    PingPongEvent,
    PingPongOut,
    ResponseOut,
    ResponseRequiredEvent,
    RetellConfig,
    RetellProtocolError,
    UpdateOnlyEvent,
    dump_outbound,
    parse_inbound,
)

MAX = 1_000_000

# Realistic samples per the Retell LLM WebSocket reference (fictional numbers/ids only).
PING_PONG_RAW = '{"interaction_type": "ping_pong", "timestamp": 1703302407333}'

CALL_DETAILS_RAW = json.dumps(
    {
        "interaction_type": "call_details",
        "call": {
            "call_type": "phone_call",
            "from_number": "+15550001234",
            "to_number": "+15550005678",
            "direction": "inbound",
            "call_id": "test-call-id-0000000000000000",
            "agent_id": "test-agent-id-000000000000000",
            "call_status": "registered",
            "metadata": {},
            "retell_llm_dynamic_variables": {"customer_name": "Test Caller"},
            "opt_out_sensitive_data_storage": True,
        },
    }
)

RESPONSE_REQUIRED_RAW = json.dumps(
    {
        "interaction_type": "response_required",
        "response_id": 3,
        "transcript": [
            {"role": "agent", "content": "Hey how can I help you?", "words": []},
            {
                "role": "user",
                "content": "Hey. How are you?",
                "words": [
                    {"word": "Hey.", "start": 4.375, "end": 4.615},
                    {"word": "How", "start": 4.615, "end": 4.855},
                    {"word": "are", "start": 4.855, "end": 5.030156},
                    {"word": "you?", "start": 5.030156, "end": 5.2053127},
                ],
            },
        ],
    }
)


class TestParseInbound:
    def test_ping_pong(self) -> None:
        event = parse_inbound(PING_PONG_RAW, max_bytes=MAX)
        assert isinstance(event, PingPongEvent)
        assert event.timestamp == 1703302407333

    def test_call_details_nested_call_object(self) -> None:
        event = parse_inbound(CALL_DETAILS_RAW, max_bytes=MAX)
        assert isinstance(event, CallDetailsEvent)
        assert event.call["call_id"] == "test-call-id-0000000000000000"
        assert event.call["retell_llm_dynamic_variables"] == {"customer_name": "Test Caller"}

    def test_response_required_words_arrays_ignored(self) -> None:
        event = parse_inbound(RESPONSE_REQUIRED_RAW, max_bytes=MAX)
        assert isinstance(event, ResponseRequiredEvent)
        assert event.response_id == 3
        assert event.is_reminder is False
        assert [u.role for u in event.transcript] == ["agent", "user"]
        assert event.transcript[1].content == "Hey. How are you?"
        assert not hasattr(event.transcript[1], "words")

    def test_reminder_required_sets_is_reminder(self) -> None:
        raw = json.dumps(
            {
                "interaction_type": "reminder_required",
                "response_id": 4,
                "transcript": [{"role": "user", "content": "..."}],
            }
        )
        event = parse_inbound(raw, max_bytes=MAX)
        assert isinstance(event, ResponseRequiredEvent)
        assert event.is_reminder is True
        assert event.response_id == 4

    def test_response_required_cannot_spoof_is_reminder(self) -> None:
        raw = json.dumps(
            {
                "interaction_type": "response_required",
                "response_id": 5,
                "transcript": [],
                "is_reminder": True,
            }
        )
        event = parse_inbound(raw, max_bytes=MAX)
        assert isinstance(event, ResponseRequiredEvent)
        assert event.is_reminder is False

    def test_update_only_with_turntaking(self) -> None:
        raw = json.dumps(
            {
                "interaction_type": "update_only",
                "transcript": [
                    {"role": "agent", "content": "Hey how can I help you?", "words": []},
                ],
                "turntaking": "agent_turn",
            }
        )
        event = parse_inbound(raw, max_bytes=MAX)
        assert isinstance(event, UpdateOnlyEvent)
        assert event.turntaking == "agent_turn"
        assert event.transcript[0].role == "agent"

    def test_update_only_without_turntaking(self) -> None:
        raw = json.dumps({"interaction_type": "update_only", "transcript": []})
        event = parse_inbound(raw, max_bytes=MAX)
        assert isinstance(event, UpdateOnlyEvent)
        assert event.turntaking is None

    def test_system_role_tolerated(self) -> None:
        raw = json.dumps(
            {
                "interaction_type": "update_only",
                "transcript": [{"role": "system", "content": "note"}],
            }
        )
        event = parse_inbound(raw, max_bytes=MAX)
        assert isinstance(event, UpdateOnlyEvent)
        assert event.transcript[0].role == "system"

    def test_bytes_input_accepted(self) -> None:
        event = parse_inbound(PING_PONG_RAW.encode("utf-8"), max_bytes=MAX)
        assert isinstance(event, PingPongEvent)


class TestSizeLimit:
    def test_oversized_valid_json_raises_before_parse(self) -> None:
        raw = json.dumps({"interaction_type": "ping_pong", "timestamp": 1, "pad": "x" * 200})
        with pytest.raises(OversizedPayloadError):
            parse_inbound(raw, max_bytes=len(raw.encode("utf-8")) - 1)

    def test_exact_limit_allowed(self) -> None:
        raw = PING_PONG_RAW
        event = parse_inbound(raw, max_bytes=len(raw.encode("utf-8")))
        assert isinstance(event, PingPongEvent)

    def test_size_measured_in_utf8_bytes_not_chars(self) -> None:
        # A non-ASCII payload is longer in bytes than in characters.
        raw = json.dumps(
            {"interaction_type": "ping_pong", "timestamp": 1, "pad": "\u00e9" * 50},
            ensure_ascii=False,
        )
        assert len(raw) < len(raw.encode("utf-8"))
        with pytest.raises(OversizedPayloadError):
            parse_inbound(raw, max_bytes=len(raw))

    def test_oversized_is_protocol_error_subclass(self) -> None:
        assert issubclass(OversizedPayloadError, RetellProtocolError)

    def test_oversized_garbage_raises_size_error_not_json_error(self) -> None:
        with pytest.raises(OversizedPayloadError):
            parse_inbound("not json at all", max_bytes=3)


class TestParseErrors:
    def test_malformed_json(self) -> None:
        with pytest.raises(RetellProtocolError):
            parse_inbound('{"interaction_type": ', max_bytes=MAX)

    def test_non_object_json(self) -> None:
        with pytest.raises(RetellProtocolError):
            parse_inbound("[1, 2, 3]", max_bytes=MAX)

    def test_unknown_interaction_type_named_but_payload_private(self) -> None:
        private = "my social security number is 000-00-0000"
        raw = json.dumps(
            {
                "interaction_type": "mystery_event",
                "transcript": [{"role": "user", "content": private}],
            }
        )
        with pytest.raises(RetellProtocolError) as excinfo:
            parse_inbound(raw, max_bytes=MAX)
        assert "mystery_event" in str(excinfo.value)
        assert private not in str(excinfo.value)

    def test_missing_interaction_type(self) -> None:
        with pytest.raises(RetellProtocolError):
            parse_inbound('{"timestamp": 1}', max_bytes=MAX)

    def test_validation_failure_names_type_not_content(self) -> None:
        private = "please charge my card ending 0000"
        raw = json.dumps(
            {
                "interaction_type": "response_required",
                "transcript": [{"role": "user", "content": private}],
                # response_id missing -> validation failure
            }
        )
        with pytest.raises(RetellProtocolError) as excinfo:
            parse_inbound(raw, max_bytes=MAX)
        message = str(excinfo.value)
        assert "response_required" in message
        assert private not in message
        # Not chained: pydantic's ValidationError embeds input values.
        assert excinfo.value.__cause__ is None

    def test_validation_error_is_not_oversized(self) -> None:
        raw = json.dumps({"interaction_type": "ping_pong", "timestamp": "soon"})
        with pytest.raises(RetellProtocolError) as excinfo:
            parse_inbound(raw, max_bytes=MAX)
        assert not isinstance(excinfo.value, OversizedPayloadError)


class TestOutbound:
    def test_dump_config(self) -> None:
        out = dump_outbound(ConfigOut(config=RetellConfig(auto_reconnect=True, call_details=True)))
        assert json.loads(out) == {
            "response_type": "config",
            "config": {
                "auto_reconnect": True,
                "call_details": True,
                "transcript_with_tool_calls": False,
            },
        }

    def test_dump_response_excludes_none_keeps_false(self) -> None:
        out = dump_outbound(
            ResponseOut(response_id=3, content="I'm doing great, ", content_complete=False)
        )
        parsed = json.loads(out)
        assert parsed == {
            "response_type": "response",
            "response_id": 3,
            "content": "I'm doing great, ",
            "content_complete": False,
        }
        assert "end_call" not in parsed
        assert "no_interruption_allowed" not in parsed

    def test_dump_response_includes_set_optionals(self) -> None:
        out = dump_outbound(
            ResponseOut(
                response_id=10,
                content="Goodbye.",
                content_complete=True,
                end_call=True,
                no_interruption_allowed=False,
            )
        )
        parsed = json.loads(out)
        assert parsed["end_call"] is True
        assert parsed["no_interruption_allowed"] is False

    def test_dump_ping_pong_echoes_timestamp(self) -> None:
        out = dump_outbound(PingPongOut(timestamp=1703302407333))
        assert json.loads(out) == {"response_type": "ping_pong", "timestamp": 1703302407333}

    def test_strict_content_complete_rejects_string(self) -> None:
        with pytest.raises(ValidationError):
            ResponseOut(response_id=1, content="hi", content_complete="true")  # type: ignore[arg-type]

    def test_strict_response_id_rejects_string(self) -> None:
        with pytest.raises(ValidationError):
            ResponseOut(response_id="1", content="hi", content_complete=True)  # type: ignore[arg-type]

    def test_strict_content_rejects_non_string(self) -> None:
        with pytest.raises(ValidationError):
            ResponseOut(response_id=1, content=123, content_complete=True)  # type: ignore[arg-type]

    def test_response_round_trip_exact_keys(self) -> None:
        out = dump_outbound(ResponseOut(response_id=7, content="ok", content_complete=True))
        parsed = json.loads(out)
        assert set(parsed) == {"response_type", "response_id", "content", "content_complete"}
        again = ResponseOut.model_validate(parsed)
        assert dump_outbound(again) == out

    def test_config_round_trip_exact_keys(self) -> None:
        out = dump_outbound(ConfigOut(config=RetellConfig(auto_reconnect=True, call_details=False)))
        parsed = json.loads(out)
        assert set(parsed) == {"response_type", "config"}
        assert set(parsed["config"]) == {
            "auto_reconnect",
            "call_details",
            "transcript_with_tool_calls",
        }
        again = ConfigOut.model_validate(parsed)
        assert dump_outbound(again) == out
