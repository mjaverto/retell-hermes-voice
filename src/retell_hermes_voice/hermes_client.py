"""Hermes Agent API client for voice turns.

Drives one phone turn as ``POST /v1/runs`` + ``GET /v1/runs/{run_id}/events`` (SSE) and
stops abandoned runs via ``POST /v1/runs/{run_id}/stop``, per the measured contract in
``hermes-contract.md``: an explicit stop cancels a run in ~0.25 s (section 2a), while
hanging up on the events stream alone leaves the run executing indefinitely (section 2c).
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator, Mapping
from typing import Any, Literal

import httpx
from pydantic import BaseModel

from .config import Settings

logger = logging.getLogger(__name__)

# Hermes writes an SSE keepalive comment after 30 s of silence (hermes-contract.md
# section 1.2). The httpx read timeout must sit safely above that interval; the real
# turn limits are enforced with asyncio deadlines instead, because every keepalive
# would otherwise reset a read timeout and mask a silent, stuck run.
_SSE_KEEPALIVE_SECONDS = 30.0

# Hard cap on the warmup turn; cold provider-client init costs tens of seconds
# (hermes-contract.md section 5.2) and must never block startup forever.
_WARMUP_CAP_SECONDS = 30.0

# Safe spoken messages. Provider/internal error text is NEVER forwarded to a caller.
SAFE_TIMEOUT_MESSAGE = "Sorry, that's taking longer than expected. Could you say that again?"
SAFE_BUSY_MESSAGE = "I'm handling too many calls right now. Please try again in a moment."
SAFE_ERROR_MESSAGE = "Sorry, I ran into a problem with that. Could you say that again?"

# Fail-open error bodies observed live (hermes-contract.md section 9, "Unknown model /
# unknown provider -- both fail open as HTTP 200"): Hermes returns a *successful* run
# whose assistant content IS the error text (with zero usage), e.g.
# "\u26a0\ufe0f Provider authentication failed: Unknown provider 'not-a-provider'. ...".
# A naive adapter would read that aloud. Any delta/done text starting with one of these
# prefixes is treated as a failed turn and replaced with a safe message.
ERROR_CONTENT_PREFIXES: tuple[str, ...] = (
    "\u26a0\ufe0f",  # warning-sign emoji prefix on provider auth failures
    "Provider authentication failed",
    "Error:",
)

# Text deltas arrive as "message.delta" on the live /v1/runs stream (hermes-contract.md
# section 1.2); "assistant.delta" / "response.output_text.delta" style names are handled
# defensively per the build contract.
_DELTA_EVENT_NAMES = frozenset({"message.delta", "assistant.delta", "response.output_text.delta"})
_TOOL_START_EVENT_NAMES = frozenset({"tool.started", "hermes.tool.progress", "subagent.start"})
_DELTA_TEXT_KEYS = ("delta", "text", "content")
_DONE_TEXT_KEYS = ("output", "text", "content")


class HermesTurnEvent(BaseModel):
    kind: Literal["delta", "tool_start", "done", "error"]
    text: str = ""


class HermesUnavailableError(Exception):
    """Hermes could not be reached at all (connection-level failure)."""


def _classify_error_prefix(text: str) -> Literal["error", "ambiguous", "clean"]:
    """Classify text against the known fail-open error prefixes.

    "ambiguous" means the text is still a proper prefix of one of the error prefixes,
    so there are not enough characters yet to rule the error out (token-level deltas
    can split a prefix across chunks).
    """
    if any(text.startswith(prefix) for prefix in ERROR_CONTENT_PREFIXES):
        return "error"
    if any(prefix.startswith(text) for prefix in ERROR_CONTENT_PREFIXES):
        return "ambiguous"
    return "clean"


def _extract_text(payload: Mapping[str, Any], keys: tuple[str, ...]) -> str:
    """Return the first string field found under ``keys``, else ""."""
    for key in keys:
        value = payload.get(key)
        if isinstance(value, str):
            return value
    return ""


def _parse_sse_json(data_lines: list[str]) -> dict[str, Any] | None:
    """Join accumulated SSE ``data:`` lines and parse them as a JSON object."""
    raw = "\n".join(data_lines).strip()
    if not raw or raw == "[DONE]":
        return None
    parsed: Any
    try:
        parsed = json.loads(raw)
    except ValueError:
        return None
    if isinstance(parsed, dict):
        return parsed
    return None


async def _next_line(lines: AsyncIterator[str]) -> str | None:
    """Advance an SSE line iterator, mapping exhaustion to None.

    A plain coroutine so the caller can bound it with ``asyncio.wait_for``.
    """
    try:
        return await lines.__anext__()
    except StopAsyncIteration:
        return None


class HermesClient:
    """One shared HTTP client for all calls; never rebuilt per turn."""

    def __init__(
        self, settings: Settings, transport: httpx.AsyncBaseTransport | None = None
    ) -> None:
        self._settings = settings
        self._client = httpx.AsyncClient(
            base_url=settings.hermes_base_url,
            headers={"Authorization": f"Bearer {settings.hermes_api_key.get_secret_value()}"},
            timeout=httpx.Timeout(
                connect=settings.hermes_connect_timeout,
                read=max(settings.hermes_turn_timeout, _SSE_KEEPALIVE_SECONDS * 2),
                write=settings.hermes_connect_timeout,
                pool=settings.hermes_connect_timeout,
            ),
            limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
            transport=transport,
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def health(self) -> bool:
        """``GET /health``: True iff HTTP 200 with ``status: ok``.

        Raises :class:`HermesUnavailableError` when Hermes cannot be reached at all.
        """
        try:
            response = await self._client.get("/health")
        except httpx.HTTPError as exc:
            raise HermesUnavailableError("Hermes API server is unreachable") from exc
        if response.status_code != 200:
            return False
        try:
            payload = response.json()
        except ValueError:
            return False
        return isinstance(payload, dict) and payload.get("status") == "ok"

    async def warmup(self) -> None:
        """Fire one tiny run with voice routing applied to warm provider clients.

        The first use of a routed provider costs tens of seconds (hermes-contract.md
        section 5.2, cold-start caveat); warming off the call path keeps the first
        real turn fast. All failures are logged and swallowed.
        """
        try:
            await asyncio.wait_for(self._drain_warmup_turn(), timeout=_WARMUP_CAP_SECONDS)
        except Exception:
            logger.warning("hermes warmup failed", exc_info=True)

    async def _drain_warmup_turn(self) -> None:
        async for _event in self.run_turn(
            session_id="rhv-warmup",
            session_key="rhv-warmup-key",
            instructions="Reply with exactly: OK",
            conversation_history=[],
            user_input="Say OK.",
        ):
            pass

    async def run_turn(
        self,
        *,
        session_id: str,
        session_key: str,
        instructions: str,
        conversation_history: list[dict[str, str]],
        user_input: str,
    ) -> AsyncIterator[HermesTurnEvent]:
        """Drive one voice turn against Hermes.

        Yields ``delta`` events as text arrives, ``tool_start`` on tool lifecycle
        events, and ``done`` at completion. Timeouts, transport failures, 429s and
        fail-open error bodies all surface as a single ``error`` event carrying a
        safe spoken message -- this generator never raises into the WS loop.
        On close/cancellation the run is always stopped unless it already reached a
        terminal event (a stop on a finished run would 404, contract section 1.4).
        """
        settings = self._settings
        loop = asyncio.get_running_loop()
        turn_deadline = loop.time() + settings.hermes_turn_timeout
        first_token_deadline = loop.time() + settings.hermes_first_token_timeout
        run_id: str | None = None
        terminal = False
        try:
            body: dict[str, Any] = {
                "input": user_input,
                "session_id": session_id,
                "instructions": instructions,
                "conversation_history": conversation_history,
            }
            if settings.voice_model is not None:
                body["model"] = settings.voice_model
            if settings.voice_provider is not None:
                body["provider"] = settings.voice_provider
            if settings.voice_reasoning_effort is not None:
                body["model_options"] = {"reasoning_effort": settings.voice_reasoning_effort}
            headers = {
                "X-Hermes-Session-Id": session_id,
                "X-Hermes-Session-Key": session_key,
            }
            try:
                created = await asyncio.wait_for(
                    self._client.post("/v1/runs", json=body, headers=headers),
                    timeout=min(first_token_deadline, turn_deadline) - loop.time(),
                )
            except TimeoutError:
                logger.warning("hermes run create timed out")
                yield HermesTurnEvent(kind="error", text=SAFE_TIMEOUT_MESSAGE)
                return
            except httpx.HTTPError:
                logger.warning("hermes run create failed: transport error")
                yield HermesTurnEvent(kind="error", text=SAFE_ERROR_MESSAGE)
                return
            if created.status_code == 429:
                # Concurrency cap shared across every agent-serving endpoint
                # (contract section 9).
                logger.warning("hermes run create rejected: concurrency limit (429)")
                yield HermesTurnEvent(kind="error", text=SAFE_BUSY_MESSAGE)
                return
            if not 200 <= created.status_code < 300:
                logger.warning("hermes run create failed: status=%s", created.status_code)
                yield HermesTurnEvent(kind="error", text=SAFE_ERROR_MESSAGE)
                return
            created_payload: Any
            try:
                created_payload = created.json()
            except ValueError:
                created_payload = None
            raw_run_id = (
                created_payload.get("run_id") if isinstance(created_payload, dict) else None
            )
            if not isinstance(raw_run_id, str) or not raw_run_id:
                logger.warning("hermes run create returned no run_id")
                yield HermesTurnEvent(kind="error", text=SAFE_ERROR_MESSAGE)
                return
            run_id = raw_run_id

            response: httpx.Response | None = None
            try:
                request = self._client.build_request("GET", f"/v1/runs/{run_id}/events")
                try:
                    response = await asyncio.wait_for(
                        self._client.send(request, stream=True),
                        timeout=min(first_token_deadline, turn_deadline) - loop.time(),
                    )
                except TimeoutError:
                    logger.warning("hermes events subscribe timed out run_id=%s", run_id)
                    yield HermesTurnEvent(kind="error", text=SAFE_TIMEOUT_MESSAGE)
                    return
                except httpx.HTTPError:
                    logger.warning("hermes events subscribe failed run_id=%s", run_id)
                    yield HermesTurnEvent(kind="error", text=SAFE_ERROR_MESSAGE)
                    return
                if response.status_code != 200:
                    logger.warning(
                        "hermes events subscribe failed run_id=%s status=%s",
                        run_id,
                        response.status_code,
                    )
                    yield HermesTurnEvent(kind="error", text=SAFE_ERROR_MESSAGE)
                    return

                lines = response.aiter_lines()
                got_first_delta = False
                decided_clean = False  # held-back text proven not to be an error body
                pending = ""  # text held back until decided_clean
                event_name = ""
                data_lines: list[str] = []
                eof = False
                while not eof:
                    deadline = (
                        turn_deadline
                        if got_first_delta
                        else min(first_token_deadline, turn_deadline)
                    )
                    remaining = deadline - loop.time()
                    if remaining <= 0:
                        logger.warning("hermes turn timed out run_id=%s", run_id)
                        yield HermesTurnEvent(kind="error", text=SAFE_TIMEOUT_MESSAGE)
                        return
                    try:
                        line = await asyncio.wait_for(_next_line(lines), timeout=remaining)
                    except TimeoutError:
                        logger.warning("hermes turn timed out run_id=%s", run_id)
                        yield HermesTurnEvent(kind="error", text=SAFE_TIMEOUT_MESSAGE)
                        return
                    except httpx.HTTPError:
                        logger.warning("hermes events stream failed run_id=%s", run_id)
                        yield HermesTurnEvent(kind="error", text=SAFE_ERROR_MESSAGE)
                        return

                    if line is None:
                        eof = True  # dispatch any final unterminated frame, then exit
                    elif line.startswith(":"):
                        continue  # SSE comment (": keepalive", ": stream closed")
                    elif line.startswith("event:"):
                        event_name = line[len("event:") :].strip()
                        continue
                    elif line.startswith("data:"):
                        chunk = line[len("data:") :]
                        data_lines.append(chunk[1:] if chunk.startswith(" ") else chunk)
                        continue
                    elif line != "":
                        continue  # unknown SSE field -> ignore

                    payload = _parse_sse_json(data_lines)
                    name = event_name
                    event_name = ""
                    data_lines = []
                    if payload is not None and not name:
                        # /v1/runs frames are bare data: frames carrying the event
                        # name in the JSON "event" field (contract section 1.2).
                        raw_name = payload.get("event")
                        if isinstance(raw_name, str):
                            name = raw_name
                    if payload is None or not name:
                        continue

                    if name in _DELTA_EVENT_NAMES:
                        got_first_delta = True
                        text = _extract_text(payload, _DELTA_TEXT_KEYS)
                        if not text:
                            continue
                        if decided_clean:
                            yield HermesTurnEvent(kind="delta", text=text)
                            continue
                        pending += text
                        verdict = _classify_error_prefix(pending)
                        if verdict == "error":
                            logger.warning(
                                "hermes fail-open error content intercepted run_id=%s", run_id
                            )
                            yield HermesTurnEvent(kind="error", text=SAFE_ERROR_MESSAGE)
                            return
                        if verdict == "clean":
                            decided_clean = True
                            yield HermesTurnEvent(kind="delta", text=pending)
                            pending = ""
                        # "ambiguous": hold back until more text arrives.
                    elif name in _TOOL_START_EVENT_NAMES:
                        if name == "hermes.tool.progress" and payload.get("status") == "completed":
                            continue  # tool end, not a start
                        # A running tool proves liveness: re-arm the first-token
                        # window so long tool phases are not killed prematurely
                        # (the session layer re-arms its filler on this event too).
                        first_token_deadline = loop.time() + settings.hermes_first_token_timeout
                        yield HermesTurnEvent(kind="tool_start")
                    elif name == "run.completed":
                        terminal = True
                        final_text = _extract_text(payload, _DONE_TEXT_KEYS)
                        if (
                            _classify_error_prefix(final_text) == "error"
                            or _classify_error_prefix(pending) == "error"
                        ):
                            logger.warning(
                                "hermes fail-open error content intercepted run_id=%s", run_id
                            )
                            yield HermesTurnEvent(kind="error", text=SAFE_ERROR_MESSAGE)
                            return
                        if pending:
                            # Held-back text turned out to be harmless; flush it.
                            yield HermesTurnEvent(kind="delta", text=pending)
                            pending = ""
                        yield HermesTurnEvent(kind="done", text=final_text)
                        return
                    elif name == "run.failed":
                        terminal = True
                        logger.warning("hermes run failed run_id=%s", run_id)
                        yield HermesTurnEvent(kind="error", text=SAFE_ERROR_MESSAGE)
                        return
                    elif name == "run.cancelled":
                        terminal = True
                        logger.info("hermes run cancelled externally run_id=%s", run_id)
                        return
                    # Unknown event names are ignored (forward-compat).

                # EOF without a terminal event: the run state is unknown.
                logger.warning("hermes events stream ended without completion run_id=%s", run_id)
                yield HermesTurnEvent(kind="error", text=SAFE_ERROR_MESSAGE)
                return
            finally:
                if response is not None:
                    try:
                        await response.aclose()
                    except Exception:
                        logger.warning("failed to close hermes events stream run_id=%s", run_id)
        finally:
            # Hanging up on the events stream alone leaves the run executing
            # indefinitely (contract section 2c); an abandoned turn MUST be stopped.
            if run_id is not None and not terminal:
                await self._stop_run(run_id)

    async def _stop_run(self, run_id: str) -> None:
        """Best-effort ``POST /v1/runs/{run_id}/stop`` bounded by hermes_stop_timeout.

        Runs inside generator finalization: every exception -- including a
        cancellation delivered while awaiting -- is swallowed so cleanup can never
        mask the original exit reason or leave the caller's teardown hanging.
        """
        try:
            await asyncio.wait_for(
                self._client.post(f"/v1/runs/{run_id}/stop"),
                timeout=self._settings.hermes_stop_timeout,
            )
        except (asyncio.CancelledError, Exception):  # deliberate blanket swallow in cleanup
            logger.warning("failed to stop hermes run run_id=%s", run_id)
