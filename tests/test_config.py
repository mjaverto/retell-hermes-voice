"""Tests for retell_hermes_voice.config."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from retell_hermes_voice.config import Settings, ToolPolicy, get_settings

REQUIRED_ENV: dict[str, str] = {
    "RHV_HERMES_BASE_URL": "http://127.0.0.1:8642",
    "RHV_HERMES_API_KEY": "test-api-key",
    "RHV_ROUTE_SECRET": "0123456789abcdef",
}


def _make_settings(monkeypatch: pytest.MonkeyPatch, **extra: str) -> Settings:
    for key, value in {**REQUIRED_ENV, **extra}.items():
        monkeypatch.setenv(key, value)
    return Settings(_env_file=None)


def test_env_round_trip(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _make_settings(monkeypatch, RHV_LISTEN_PORT="9001")
    assert settings.hermes_base_url == "http://127.0.0.1:8642"
    assert settings.hermes_api_key.get_secret_value() == "test-api-key"
    assert settings.route_secret.get_secret_value() == "0123456789abcdef"
    assert settings.listen_port == 9001
    assert settings.listen_host == "127.0.0.1"
    assert settings.session_retention == "none"


def test_secrets_not_in_repr(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _make_settings(monkeypatch)
    assert "test-api-key" not in repr(settings)
    assert "0123456789abcdef" not in repr(settings)


def test_route_secret_too_short_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValidationError):
        _make_settings(monkeypatch, RHV_ROUTE_SECRET="tooshort")


def test_allowed_callers_comma_parsing(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _make_settings(monkeypatch, RHV_ALLOWED_CALLERS="+15551234567, +15559876543")
    assert settings.allowed_callers == ["+15551234567", "+15559876543"]


def test_allowed_callers_json_parsing(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _make_settings(monkeypatch, RHV_ALLOWED_CALLERS='["+15551234567", "+15559876543"]')
    assert settings.allowed_callers == ["+15551234567", "+15559876543"]


def test_allowed_callers_empty_string_disables_allowlist(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _make_settings(monkeypatch, RHV_ALLOWED_CALLERS="")
    assert settings.allowed_callers == []


@pytest.mark.parametrize(
    "bad",
    ["banana", "15551234567", "+05551234567", "+1555", "+1555123456789012345"],
)
def test_allowed_callers_rejects_non_e164(monkeypatch: pytest.MonkeyPatch, bad: str) -> None:
    with pytest.raises(ValidationError):
        _make_settings(monkeypatch, RHV_ALLOWED_CALLERS=bad)


def test_filler_phrases_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _make_settings(monkeypatch)
    assert len(settings.filler_phrases) >= 6
    assert len(set(settings.filler_phrases)) == len(settings.filler_phrases)
    assert all(phrase.strip() for phrase in settings.filler_phrases)


def test_filler_phrases_comma_override(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _make_settings(monkeypatch, RHV_FILLER_PHRASES="One moment., Let me check.")
    assert settings.filler_phrases == ["One moment.", "Let me check."]


def test_filler_phrases_must_not_be_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValidationError):
        _make_settings(monkeypatch, RHV_FILLER_PHRASES="")


def test_tool_policy_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _make_settings(monkeypatch)
    assert settings.tool_policy == ToolPolicy()
    assert settings.tool_policy.enabled_tools == []
    assert settings.tool_policy.max_tool_calls_per_turn == 3


def test_tool_policy_nested_env(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _make_settings(monkeypatch, RHV_TOOL_POLICY__MAX_TOOL_CALLS_PER_TURN="5")
    assert settings.tool_policy.max_tool_calls_per_turn == 5


def test_get_settings_is_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    for key, value in REQUIRED_ENV.items():
        monkeypatch.setenv(key, value)
    get_settings.cache_clear()
    try:
        first = get_settings()
        second = get_settings()
        assert first is second
        assert first.hermes_base_url == REQUIRED_ENV["RHV_HERMES_BASE_URL"]
    finally:
        get_settings.cache_clear()
