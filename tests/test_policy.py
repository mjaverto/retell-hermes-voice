"""Tests for retell_hermes_voice.policy."""

from __future__ import annotations

import pytest

from retell_hermes_voice.policy import (
    CallerPolicy,
    derive_session_ids,
    map_transcript,
    truncate_transcript,
)
from retell_hermes_voice.retell_events import Utterance


def _u(role: str, content: str) -> Utterance:
    return Utterance.model_validate({"role": role, "content": content})


class TestCallerPolicy:
    def test_rejects_invalid_allowlist_entries(self) -> None:
        with pytest.raises(ValueError, match="E.164"):
            CallerPolicy(["not-a-number"])

    def test_empty_allowlist_not_enforced_allows_all(self) -> None:
        policy = CallerPolicy([])
        assert policy.enforced is False
        assert policy.is_allowed("+15551234567") is True
        assert policy.is_allowed(None) is True
        assert policy.is_allowed("garbage") is True

    def test_enforced_allows_listed_number(self) -> None:
        policy = CallerPolicy(["+15551234567", "+15559876543"])
        assert policy.enforced is True
        assert policy.is_allowed("+15551234567") is True
        assert policy.is_allowed("+15559876543") is True

    def test_enforced_denies_unlisted_number(self) -> None:
        policy = CallerPolicy(["+15551234567"])
        assert policy.is_allowed("+15550000000") is False

    def test_enforced_denies_missing_number(self) -> None:
        policy = CallerPolicy(["+15551234567"])
        assert policy.is_allowed(None) is False

    @pytest.mark.parametrize("bad", ["", "banana", "15551234567", "+0555123456", "+1"])
    def test_enforced_denies_malformed_number(self, bad: str) -> None:
        policy = CallerPolicy(["+15551234567"])
        assert policy.is_allowed(bad) is False


class TestDeriveSessionIds:
    def test_deterministic(self) -> None:
        assert derive_session_ids("call_abc123") == derive_session_ids("call_abc123")

    def test_id_and_key_distinct(self) -> None:
        session_id, session_key = derive_session_ids("call_abc123")
        assert session_id != session_key

    def test_distinct_across_calls(self) -> None:
        assert derive_session_ids("call_one") != derive_session_ids("call_two")

    def test_prefixes_and_hash_length(self) -> None:
        session_id, session_key = derive_session_ids("call_abc123")
        assert session_id.startswith("rhv-")
        assert session_key.startswith("rhv-key-")
        assert len(session_id) == len("rhv-") + 24
        assert len(session_key) == len("rhv-key-") + 24

    def test_raw_call_id_not_embedded(self) -> None:
        call_id = "call_super_secret_id"
        session_id, session_key = derive_session_ids(call_id)
        assert call_id not in session_id
        assert call_id not in session_key


class TestTruncateTranscript:
    def test_empty_transcript(self) -> None:
        assert truncate_transcript([], 10) == []

    def test_under_limit_untouched(self) -> None:
        transcript = [_u("user", "hi"), _u("agent", "hello")]
        assert truncate_transcript(transcript, 5) == transcript

    def test_keeps_most_recent(self) -> None:
        transcript = [_u("user", str(i)) for i in range(6)]
        result = truncate_transcript(transcript, 2)
        assert [u.content for u in result] == ["4", "5"]

    def test_always_keeps_last_utterance(self) -> None:
        transcript = [_u("user", "old"), _u("agent", "latest")]
        result = truncate_transcript(transcript, 0)
        assert [u.content for u in result] == ["latest"]


class TestMapTranscript:
    def test_role_mapping(self) -> None:
        transcript = [_u("user", "hi there"), _u("agent", "hello caller")]
        assert map_transcript(transcript) == [
            {"role": "user", "content": "hi there"},
            {"role": "assistant", "content": "hello caller"},
        ]

    def test_system_entries_dropped(self) -> None:
        transcript = [_u("system", "internal note"), _u("user", "hi")]
        assert map_transcript(transcript) == [{"role": "user", "content": "hi"}]

    def test_empty_content_skipped(self) -> None:
        transcript = [_u("user", ""), _u("agent", "   "), _u("user", "real")]
        assert map_transcript(transcript) == [{"role": "user", "content": "real"}]

    def test_unintelligible_audio_passed_through(self) -> None:
        transcript = [_u("user", "(unintelligible audio)")]
        assert map_transcript(transcript) == [{"role": "user", "content": "(unintelligible audio)"}]
