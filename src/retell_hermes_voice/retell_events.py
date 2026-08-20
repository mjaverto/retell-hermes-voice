"""Retell AI Custom LLM wire types (pydantic v2).

Inbound events are discriminated by ``interaction_type``; outbound events by
``response_type``. See docs/INTERFACES.md for the fixed cross-module contract.

Privacy: parse errors NEVER include payload content in their messages —
transcripts are private. Only the ``interaction_type`` value may appear.
"""

from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    StrictBool,
    StrictInt,
    StrictStr,
    ValidationError,
)


class RetellProtocolError(Exception):
    """A frame from Retell could not be parsed into a known inbound event."""


class OversizedPayloadError(RetellProtocolError):
    """An inbound frame exceeded the configured byte limit."""


class Utterance(BaseModel):
    """One transcript entry. Extra fields (e.g. ``words`` timing arrays) are dropped."""

    model_config = ConfigDict(extra="ignore")

    role: Literal["agent", "user", "system"]  # "system" tolerated for forward-compat
    content: str


class CallDetailsEvent(BaseModel):
    """``interaction_type == "call_details"``: the full call object from Register Call."""

    model_config = ConfigDict(extra="ignore")

    call: dict[str, Any]


class ResponseRequiredEvent(BaseModel):
    """``response_required`` or ``reminder_required``: Retell is waiting to speak."""

    model_config = ConfigDict(extra="ignore")

    response_id: int
    transcript: list[Utterance]
    is_reminder: bool = False


class UpdateOnlyEvent(BaseModel):
    """``interaction_type == "update_only"``: transcript/turntaking updates, no reply."""

    model_config = ConfigDict(extra="ignore")

    transcript: list[Utterance]
    turntaking: Literal["agent_turn", "user_turn"] | None = None


class PingPongEvent(BaseModel):
    """``interaction_type == "ping_pong"``: keepalive; must be echoed within 5s."""

    model_config = ConfigDict(extra="ignore")

    timestamp: int


InboundEvent = CallDetailsEvent | ResponseRequiredEvent | UpdateOnlyEvent | PingPongEvent


def parse_inbound(raw: str | bytes, *, max_bytes: int) -> InboundEvent:
    """Parse one inbound Retell frame.

    Raises :class:`OversizedPayloadError` (before any JSON parsing) when the
    frame's byte length exceeds ``max_bytes``; :class:`RetellProtocolError` on
    malformed JSON, unknown ``interaction_type``, or failed validation. Error
    messages never contain payload content.
    """
    data = raw.encode("utf-8") if isinstance(raw, str) else raw
    if len(data) > max_bytes:
        raise OversizedPayloadError(
            f"inbound frame of {len(data)} bytes exceeds limit of {max_bytes} bytes"
        )
    try:
        payload = json.loads(data)
    except (json.JSONDecodeError, UnicodeDecodeError):
        raise RetellProtocolError("inbound frame is not valid JSON") from None
    if not isinstance(payload, dict):
        raise RetellProtocolError("inbound frame is not a JSON object")
    interaction_type = payload.get("interaction_type")
    try:
        if interaction_type == "call_details":
            return CallDetailsEvent.model_validate(payload)
        if interaction_type in ("response_required", "reminder_required"):
            return ResponseRequiredEvent.model_validate(
                {**payload, "is_reminder": interaction_type == "reminder_required"}
            )
        if interaction_type == "update_only":
            return UpdateOnlyEvent.model_validate(payload)
        if interaction_type == "ping_pong":
            return PingPongEvent.model_validate(payload)
    except ValidationError:
        # Deliberately not chained: pydantic error details embed input values,
        # and transcripts must never leak into logs or tracebacks.
        raise RetellProtocolError(
            f"invalid payload for interaction_type {interaction_type!r}"
        ) from None
    raise RetellProtocolError(f"unknown interaction_type {interaction_type!r}")


class RetellConfig(BaseModel):
    """Payload of the ``config`` event."""

    auto_reconnect: bool
    call_details: bool
    transcript_with_tool_calls: bool = False


class ConfigOut(BaseModel):
    """``config`` event: must be the first frame sent, if sent at all."""

    response_type: Literal["config"] = "config"
    config: RetellConfig


class ResponseOut(BaseModel):
    """``response`` event: agent speech for a specific ``response_id``.

    ``response_id``/``content``/``content_complete`` use strict types: Retell
    silently drops a response event when ``content_complete`` isn't a boolean,
    ``content`` isn't a string, or ``response_id`` isn't an integer.
    """

    response_type: Literal["response"] = "response"
    response_id: StrictInt
    content: StrictStr
    content_complete: StrictBool
    end_call: bool | None = None
    no_interruption_allowed: bool | None = None


class PingPongOut(BaseModel):
    """``ping_pong`` reply: echo the inbound timestamp unmodified."""

    response_type: Literal["ping_pong"] = "ping_pong"
    timestamp: int


def dump_outbound(event: ConfigOut | ResponseOut | PingPongOut) -> str:
    """Serialize an outbound event to JSON, omitting fields that are ``None``."""
    return event.model_dump_json(exclude_none=True)
