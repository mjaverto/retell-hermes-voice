"""Tests for retell_hermes_voice.logredact."""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Iterator

import pytest

from retell_hermes_voice import logredact
from retell_hermes_voice.config import Settings
from retell_hermes_voice.logredact import (
    SECRET_ENV_NAMES,
    RedactionFilter,
    configure_logging,
    redact_phone,
)

FAKE_SECRET = "sekret-value-123456"


def _record(msg: str, args: tuple[object, ...] = ()) -> logging.LogRecord:
    return logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg=msg,
        args=args or None,
        exc_info=None,
    )


def test_secret_env_names() -> None:
    assert SECRET_ENV_NAMES == ("RHV_HERMES_API_KEY", "RHV_ROUTE_SECRET")


class TestRedactPhone:
    def test_masks_middle_digits(self) -> None:
        assert redact_phone("+13475551234") == "+1******1234"

    def test_longer_number(self) -> None:
        assert redact_phone("+441632960123") == "+4*******0123"

    def test_non_e164_fully_masked(self) -> None:
        assert redact_phone("banana") == "******"


class TestRedactionFilter:
    def test_scrubs_secret_and_phone_from_formatted_message(self) -> None:
        filt = RedactionFilter([FAKE_SECRET])
        record = _record("key=%s caller=%s", (FAKE_SECRET, "+13475551234"))
        assert filt.filter(record) is True
        message = record.getMessage()
        assert FAKE_SECRET not in message
        assert "[REDACTED]" in message
        assert "+13475551234" not in message
        assert "+1******1234" in message

    def test_plain_message_untouched(self) -> None:
        filt = RedactionFilter([FAKE_SECRET])
        record = _record("call started ok")
        filt.filter(record)
        assert record.getMessage() == "call started ok"

    def test_empty_secrets_ignored(self) -> None:
        filt = RedactionFilter(["", FAKE_SECRET])
        record = _record(f"value={FAKE_SECRET}")
        filt.filter(record)
        assert "[REDACTED]" in record.getMessage()

    def test_bad_percent_format_does_not_raise_and_still_scrubs(self) -> None:
        filt = RedactionFilter([FAKE_SECRET])
        record = _record(f"val={FAKE_SECRET} n=%d", ("not-a-number",))
        assert filt.filter(record) is True
        message = record.getMessage()
        assert FAKE_SECRET not in message
        assert "[REDACTED]" in message
        assert record.args is None

    def test_ws_path_secret_and_call_id_rewritten(self) -> None:
        filt = RedactionFilter([])
        record = _record('connection accepted "/llm-websocket/supersecretvalue123456/call-abc"')
        filt.filter(record)
        message = record.getMessage()
        assert "supersecretvalue123456" not in message
        assert "call-abc" not in message
        expected_ref = hashlib.sha256(b"call-abc").hexdigest()[:12]
        assert f"/llm-websocket/[REDACTED]/{expected_ref}" in message


def _settings() -> Settings:
    return Settings(
        hermes_base_url="http://127.0.0.1:8642",
        hermes_api_key="test-api-key",
        route_secret="0123456789abcdef",
        _env_file=None,
    )


@pytest.fixture()
def clean_root_logger() -> Iterator[logging.Logger]:
    root = logging.getLogger()
    before = list(root.handlers)
    level = root.level
    yield root
    for handler in list(root.handlers):
        if handler not in before:
            root.removeHandler(handler)
    root.setLevel(level)
    if hasattr(root, logredact._CONFIGURED_ATTR):
        delattr(root, logredact._CONFIGURED_ATTR)


def test_configure_logging_idempotent(clean_root_logger: logging.Logger) -> None:
    root = clean_root_logger
    before = list(root.handlers)
    settings = _settings()

    configure_logging(settings)
    added = [h for h in root.handlers if h not in before]
    assert len(added) == 1
    assert any(isinstance(f, RedactionFilter) for f in added[0].filters)
    assert root.level == logging.INFO

    configure_logging(settings)
    assert [h for h in root.handlers if h not in before] == added
