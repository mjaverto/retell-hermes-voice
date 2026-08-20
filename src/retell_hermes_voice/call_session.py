"""Per-call orchestration: one :class:`CallSession` owns one Retell WebSocket."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import secrets
import time
from collections.abc import AsyncGenerator, Awaitable, Callable
from typing import Any, Literal, cast

from .config import Settings
from .hermes_client import HermesClient, HermesTurnEvent
from .logredact import redact_phone
from .policy import CallerPolicy, derive_session_ids, map_transcript, truncate_transcript
from .retell_events import (
    CallDetailsEvent,
    InboundEvent,
    PingPongEvent,
    PingPongOut,
    ResponseOut,
    ResponseRequiredEvent,
    UpdateOnlyEvent,
    dump_outbound,
)
from .speech import (
    DeltaSanitizer,
    FillerPicker,
    build_instructions,
    build_opening,
    sanitize_spoken,
)

logger = logging.getLogger(__name__)

DENIED_LINE = "Sorry, this number isn't available. Goodbye."
APOLOGY_LINE = "Sorry, I'm having trouble right now. Could you say that again?"


class CallSession:
    """Owns one WebSocket = one call.

    Handles greeting/allowlist on ``call_details``, ping_pong echo, per-turn Hermes
    streaming with staleness guards, dead-air filler phrases, and idempotent close.
    """

    def __init__(
        self,
        *,
        call_id: str,
        settings: Settings,
        hermes: HermesClient,
        send: Callable[[str], Awaitable[None]],
    ) -> None:
        self._call_id = call_id
        self._settings = settings
        self._hermes = hermes
        self._send = send
        self._policy = CallerPolicy(settings.allowed_callers)
        self._filler = FillerPicker(settings.filler_phrases)
        if settings.session_retention == "hermes":
            self.session_id, self._session_key = derive_session_ids(call_id)
        else:
            # "none": per-call random ids; nothing links the Hermes session to the call.
            self.session_id = f"rhv-{secrets.token_hex(12)}"
            self._session_key = f"rhv-key-{secrets.token_hex(12)}"
        self.latest_response_id: int = -1
        self.turns: int = 0
        self.outcome: str = "open"
        self._turn_task: asyncio.Task[None] | None = None
        self._greeted = False
        self._denied = False
        self._closed = False
        self._direction: Literal["inbound", "outbound"] = "inbound"
        self._from_number: str | None = None
        self._tool_seconds_used: float = 0.0
        self._tool_budget_warned = False

    async def on_event(self, event: InboundEvent) -> None:
        """Dispatch one inbound Retell event. Never blocks on turn work."""
        if self._closed:
            return
        if isinstance(event, PingPongEvent):
            await self._send_event(PingPongOut(timestamp=event.timestamp))
        elif isinstance(event, CallDetailsEvent):
            await self._handle_call_details(event)
        elif isinstance(event, ResponseRequiredEvent):
            await self._handle_response_required(event)
        elif isinstance(event, UpdateOnlyEvent):
            # Ignored: the transcript arrives fresh on every response_required.
            return

    async def close(self) -> None:
        """Cancel any in-flight turn (stopping the Hermes run). Idempotent."""
        if self._closed:
            return
        self._closed = True
        if self.outcome == "open":
            self.outcome = "closed"
        await self._cancel_turn_task()

    async def _handle_call_details(self, event: CallDetailsEvent) -> None:
        call: dict[str, Any] = event.call if isinstance(event.call, dict) else {}
        self._direction = "outbound" if call.get("direction") == "outbound" else "inbound"
        raw_from = call.get("from_number")
        self._from_number = raw_from if isinstance(raw_from, str) else None
        if self._policy.enforced and not self._policy.is_allowed(self._from_number):
            self._denied = True
            self.outcome = "denied"
            logger.info(
                "call denied call_id=%s from=%s outcome=denied",
                self._call_id,
                redact_phone(self._from_number) if self._from_number else "unknown",
            )
            await self._send_event(
                ResponseOut(
                    response_id=0, content=DENIED_LINE, content_complete=True, end_call=True
                )
            )
            return
        if self._greeted:
            return
        opening = build_opening(
            self._settings,
            direction=self._direction,
            dynamic_variables=self._dynamic_variables(call),
        )
        await self._send_event(ResponseOut(response_id=0, content=opening, content_complete=True))
        self._greeted = True

    async def _handle_response_required(self, event: ResponseRequiredEvent) -> None:
        if self._denied or event.response_id <= self.latest_response_id:
            return
        self.latest_response_id = event.response_id
        self.turns += 1
        received_at = time.monotonic()
        await self._cancel_turn_task()
        self._turn_task = asyncio.create_task(
            self._run_turn(event, received_at), name=f"rhv-turn-{event.response_id}"
        )

    async def _cancel_turn_task(self) -> None:
        task = self._turn_task
        self._turn_task = None
        if task is None or task.done():
            return
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    async def _run_turn(self, event: ResponseRequiredEvent, received_at: float) -> None:
        response_id = event.response_id
        settings = self._settings
        loop = asyncio.get_running_loop()
        outcome = "stale"
        ttfb_ms: int | None = None
        agen: AsyncGenerator[HermesTurnEvent, None] | None = None
        next_task: asyncio.Task[HermesTurnEvent] | None = None
        try:
            transcript = truncate_transcript(event.transcript, settings.max_transcript_utterances)
            history = map_transcript(transcript)
            user_input = ""
            if history and history[-1]["role"] == "user":
                # The trailing user utterance is the turn input, not history.
                user_input = history[-1]["content"]
                history = history[:-1]
            agen = cast(
                AsyncGenerator[HermesTurnEvent, None],
                self._hermes.run_turn(
                    session_id=self.session_id,
                    session_key=self._session_key,
                    instructions=build_instructions(settings, direction=self._direction),
                    conversation_history=history,
                    user_input=user_input,
                ),
            )
            sanitizer = DeltaSanitizer()
            filler_after = settings.filler_after_seconds
            filler_deadline: float | None = loop.time() + filler_after
            last_emit_at: float | None = None
            emitted_delta = False
            tool_started_at: float | None = None
            while True:
                if next_task is None:
                    next_task = asyncio.ensure_future(anext(agen))
                timeout: float | None = None
                if filler_deadline is not None:
                    timeout = max(0.0, filler_deadline - loop.time())
                done, _pending = await asyncio.wait({next_task}, timeout=timeout)
                now = loop.time()
                if not done:
                    # Dead air: the filler window elapsed before the next Hermes event.
                    filler_deadline = None  # re-armed only by a later tool_start
                    if response_id != self.latest_response_id:
                        return
                    if last_emit_at is None or now - last_emit_at >= filler_after:
                        filler = self._filler.pick() + " "
                        if not await self._emit(response_id, filler, complete=False):
                            return
                        last_emit_at = now
                    continue
                try:
                    turn_event = next_task.result()
                except StopAsyncIteration:
                    turn_event = HermesTurnEvent(kind="done")
                next_task = None
                if tool_started_at is not None:
                    self._note_tool_time(now - tool_started_at)
                    tool_started_at = None
                if turn_event.kind == "delta":
                    filler_deadline = None
                    text = sanitizer.feed(turn_event.text)
                    if text:
                        if not await self._emit(response_id, text, complete=False):
                            return
                        last_emit_at = now
                        if not emitted_delta:
                            emitted_delta = True
                            ttfb_ms = int((now - received_at) * 1000)
                            logger.info(
                                "turn first_delta call=%s turn=%d ttfb_ms=%d",
                                self.session_id,
                                response_id,
                                ttfb_ms,
                            )
                elif turn_event.kind == "tool_start":
                    tool_started_at = now
                    filler_deadline = now + filler_after
                elif turn_event.kind == "done":
                    remainder = sanitizer.flush()
                    if not emitted_delta and not remainder and turn_event.text:
                        # Hermes delivered final text without streaming any deltas.
                        remainder = sanitize_spoken(turn_event.text)
                    if not await self._emit(response_id, remainder, complete=True):
                        return
                    outcome = "ok"
                    return
                else:  # error: text is a safe, generic message by contract
                    safe_text = turn_event.text or APOLOGY_LINE
                    if not await self._emit(response_id, safe_text, complete=True):
                        return
                    outcome = "error"
                    return
        except asyncio.CancelledError:
            outcome = "superseded"
            raise
        except Exception:
            outcome = "failed"
            logger.exception("turn failed call=%s turn=%d", self.session_id, response_id)
            if response_id == self.latest_response_id and not self._closed:
                with contextlib.suppress(Exception):
                    await self._send_event(
                        ResponseOut(
                            response_id=response_id, content=APOLOGY_LINE, content_complete=True
                        )
                    )
        finally:
            if next_task is not None and not next_task.done():
                next_task.cancel()
            if next_task is not None:
                with contextlib.suppress(asyncio.CancelledError, StopAsyncIteration, Exception):
                    await next_task
            if agen is not None:
                # Mandatory: closing the generator triggers the Hermes run stop.
                with contextlib.suppress(Exception):
                    await agen.aclose()
            total_ms = int((time.monotonic() - received_at) * 1000)
            logger.info(
                "turn end call=%s turn=%d ttfb_ms=%s total_ms=%d outcome=%s",
                self.session_id,
                response_id,
                ttfb_ms if ttfb_ms is not None else "-",
                total_ms,
                outcome,
            )

    async def _emit(self, response_id: int, content: str, *, complete: bool) -> bool:
        """Send a response frame unless the turn went stale; returns True when sent."""
        if self._closed or response_id != self.latest_response_id:
            return False
        await self._send_event(
            ResponseOut(response_id=response_id, content=content, content_complete=complete)
        )
        return True

    async def _send_event(self, event: ResponseOut | PingPongOut) -> None:
        await self._send(dump_outbound(event))

    def _note_tool_time(self, elapsed: float) -> None:
        self._tool_seconds_used += elapsed
        budget = self._settings.tool_policy.max_tool_seconds_per_call
        if not self._tool_budget_warned and self._tool_seconds_used > budget:
            self._tool_budget_warned = True
            logger.warning(
                "tool budget exceeded call=%s used_s=%.1f budget_s=%.1f",
                self.session_id,
                self._tool_seconds_used,
                budget,
            )

    @staticmethod
    def _dynamic_variables(call: dict[str, Any]) -> dict[str, str] | None:
        raw = call.get("retell_llm_dynamic_variables")
        if not isinstance(raw, dict):
            return None
        return {str(key): str(value) for key, value in raw.items()}
