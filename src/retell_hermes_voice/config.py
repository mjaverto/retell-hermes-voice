"""Runtime settings for the Retell <-> Hermes voice adapter."""

from __future__ import annotations

import json
import re
from functools import lru_cache
from typing import Annotated, Literal

from pydantic import BaseModel, SecretStr, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

_E164_RE = re.compile(r"^\+[1-9]\d{6,14}$")

_DEFAULT_FILLER_PHRASES: tuple[str, ...] = (
    "I have that information right here, give me a second.",
    "Let me pull that up for you.",
    "One moment while I check.",
    "Just a second, looking now.",
    "Give me a moment to find that.",
    "Hold on, checking that for you.",
    "Let me take a quick look.",
    "Bear with me one second.",
)


def _split_csv(value: object) -> object:
    """Allow list fields to be set from comma-separated env strings (or JSON arrays)."""
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if text.startswith("["):
            return json.loads(text)
        return [item.strip() for item in text.split(",") if item.strip()]
    return value


class ToolPolicy(BaseModel):
    """Which Hermes tools voice turns may use, and their budgets."""

    enabled_tools: list[str] = []
    confirm_tools: list[str] = []
    max_tool_calls_per_turn: int = 3
    max_tool_seconds_per_call: float = 60.0


class Settings(BaseSettings):
    """Adapter configuration, loaded from RHV_-prefixed env vars and an optional .env file."""

    model_config = SettingsConfigDict(
        env_prefix="RHV_",
        env_file=".env",
        case_sensitive=False,
        env_nested_delimiter="__",
    )

    # hermes connection
    hermes_base_url: str
    hermes_api_key: SecretStr
    route_secret: SecretStr
    listen_host: str = "127.0.0.1"
    listen_port: int = 8765

    # caller policy (E.164; empty list = allowlist DISABLED: allow all, warn)
    allowed_callers: Annotated[list[str], NoDecode] = []

    # voice behavior
    greeting: str = "Hi, this is the assistant. How can I help?"
    outbound_opening: str = (
        "Hi, this is {assistant_name}, {principal}'s assistant. "
        "Just so you know, this call may be recorded. {purpose}"
    )
    assistant_name: str = "the assistant"
    principal: str = "the operator"
    filler_phrases: Annotated[list[str], NoDecode] = list(_DEFAULT_FILLER_PHRASES)
    filler_after_seconds: float = 1.5

    # hermes routing
    voice_model: str | None = None
    voice_provider: str | None = None
    voice_reasoning_effort: str | None = "low"
    warmup_on_start: bool = True

    # timeouts / limits
    hermes_connect_timeout: float = 5.0
    hermes_first_token_timeout: float = 15.0
    hermes_turn_timeout: float = 60.0
    hermes_stop_timeout: float = 3.0
    max_concurrent_calls: int = 5
    max_transcript_utterances: int = 200
    max_ws_message_bytes: int = 1_000_000
    session_retention: Literal["none", "hermes"] = "none"
    tool_policy: ToolPolicy = ToolPolicy()
    log_transcripts: bool = False

    @field_validator("allowed_callers", "filler_phrases", mode="before")
    @classmethod
    def _parse_csv(cls, value: object) -> object:
        return _split_csv(value)

    @field_validator("route_secret")
    @classmethod
    def _check_route_secret(cls, value: SecretStr) -> SecretStr:
        if len(value.get_secret_value()) < 16:
            raise ValueError("route_secret must be at least 16 characters long")
        return value

    @field_validator("allowed_callers")
    @classmethod
    def _check_allowed_callers(cls, value: list[str]) -> list[str]:
        for number in value:
            if _E164_RE.fullmatch(number) is None:
                raise ValueError("allowed_callers entries must be E.164 numbers like +15551234567")
        return value

    @field_validator("filler_phrases")
    @classmethod
    def _check_filler_phrases(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("filler_phrases must not be empty")
        return value


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide cached settings instance."""
    return Settings()  # type: ignore[call-arg]  # required fields come from the environment
