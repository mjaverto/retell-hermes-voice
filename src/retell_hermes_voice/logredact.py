"""Logging setup with redaction of secrets and phone numbers."""

from __future__ import annotations

import hashlib
import logging
import re
import sys
from collections.abc import Iterable

from retell_hermes_voice.config import Settings

SECRET_ENV_NAMES: tuple[str, ...] = ("RHV_HERMES_API_KEY", "RHV_ROUTE_SECRET")

_E164_RE = re.compile(r"\+[1-9]\d{6,14}")
_WS_PATH_RE = re.compile(r"/llm-websocket/([^/\s\"']+)/([^/\s?#\"']+)")
_CONFIGURED_ATTR = "_rhv_logging_configured"


def redact_phone(number: str) -> str:
    """Mask the middle digits of an E.164 number: '+13475551234' -> '+1******1234'."""
    if _E164_RE.fullmatch(number) is None:
        return "*" * len(number)
    return number[:2] + "*" * (len(number) - 6) + number[-4:]


class RedactionFilter(logging.Filter):
    """Replaces occurrences of secret values and E.164-looking numbers in log records.

    The record message is formatted eagerly (``record.getMessage()``) and rewritten so
    downstream handlers never see raw secrets or full phone numbers. WebSocket paths
    of the form ``/llm-websocket/<route-secret>/<call-id>`` (e.g. uvicorn's own
    "connection accepted" line, which bypasses app-level hashing) are rewritten too:
    the route secret becomes ``[REDACTED]`` and the call id its short sha256 ref.
    """

    def __init__(self, secrets: Iterable[str]) -> None:
        super().__init__()
        self._secrets: tuple[str, ...] = tuple(s for s in secrets if s)

    def _redact(self, text: str) -> str:
        for secret in self._secrets:
            text = text.replace(secret, "[REDACTED]")
        text = _WS_PATH_RE.sub(
            lambda match: (
                "/llm-websocket/[REDACTED]/"
                + hashlib.sha256(match.group(2).encode()).hexdigest()[:12]
            ),
            text,
        )
        return _E164_RE.sub(lambda match: redact_phone(match.group(0)), text)

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
        except Exception:  # bad %-format call: scrub what we have, never raise
            message = str(record.msg)
        record.msg = self._redact(message)
        record.args = None
        return True


def configure_logging(settings: Settings) -> None:
    """Install key=value stderr logging with redaction on the root logger. Idempotent."""
    root = logging.getLogger()
    if getattr(root, _CONFIGURED_ATTR, False):
        return
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        logging.Formatter("ts=%(asctime)s level=%(levelname)s logger=%(name)s msg=%(message)s")
    )
    handler.addFilter(
        RedactionFilter(
            (
                settings.hermes_api_key.get_secret_value(),
                settings.route_secret.get_secret_value(),
            )
        )
    )
    root.addHandler(handler)
    root.setLevel(logging.INFO)
    setattr(root, _CONFIGURED_ATTR, True)
