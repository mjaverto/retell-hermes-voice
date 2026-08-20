"""Caller allowlisting, session id derivation, and transcript shaping."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence

from retell_hermes_voice.retell_events import Utterance

_E164_RE = re.compile(r"^\+[1-9]\d{6,14}$")

_ROLE_MAP: dict[str, str] = {"agent": "assistant", "user": "user"}


class CallerPolicy:
    """Deny-by-default caller allowlist; disabled (allow all) when the list is empty."""

    def __init__(self, allowed: Sequence[str]) -> None:
        for number in allowed:
            if _E164_RE.fullmatch(number) is None:
                raise ValueError("allowlist entries must be E.164 numbers like +15551234567")
        self._allowed: frozenset[str] = frozenset(allowed)

    @property
    def enforced(self) -> bool:
        return bool(self._allowed)

    def is_allowed(self, from_number: str | None) -> bool:
        if not self.enforced:
            return True
        if from_number is None or _E164_RE.fullmatch(from_number) is None:
            return False
        return from_number in self._allowed


def derive_session_ids(call_id: str) -> tuple[str, str]:
    """(session_id, session_key) derived from the Retell call id, never from phone numbers."""
    session_id = "rhv-" + hashlib.sha256(call_id.encode("utf-8")).hexdigest()[:24]
    session_key = "rhv-key-" + hashlib.sha256(("k" + call_id).encode("utf-8")).hexdigest()[:24]
    return session_id, session_key


def truncate_transcript(transcript: list[Utterance], max_len: int) -> list[Utterance]:
    """Keep the most recent utterances; always keep at least the last one."""
    if not transcript:
        return []
    return transcript[-max(max_len, 1) :]


def map_transcript(transcript: list[Utterance]) -> list[dict[str, str]]:
    """Retell roles -> Hermes chat roles; system entries and empty content are dropped."""
    mapped: list[dict[str, str]] = []
    for utterance in transcript:
        role = _ROLE_MAP.get(utterance.role)
        if role is None or not utterance.content.strip():
            continue
        mapped.append({"role": role, "content": utterance.content})
    return mapped
