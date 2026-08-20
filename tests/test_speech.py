"""Unit tests for retell_hermes_voice.speech."""

from __future__ import annotations

import random
import re
import time
from typing import Any

import pytest

from retell_hermes_voice.config import Settings, ToolPolicy
from retell_hermes_voice.speech import (
    MAX_SPOKEN_INPUT,
    VOICE_SYSTEM_PROMPT,
    DeltaSanitizer,
    FillerPicker,
    build_instructions,
    build_opening,
    sanitize_spoken,
)


def make_settings(**overrides: Any) -> Settings:
    kwargs: dict[str, Any] = {
        "hermes_base_url": "http://127.0.0.1:9",
        "hermes_api_key": "unit-test-key",
        "route_secret": "unit-test-secret-0123456789",
        **overrides,
    }
    return Settings(**kwargs)


GNARLY = (
    "# Report\n\n"
    "Hello **world**, this is _important_ and __very__ *nice*.\n\n"
    "---\n\n"
    "- first point\n"
    "- second point\n\n"
    "1. one\n"
    "2. two\n\n"
    "| Name | Age |\n| --- | --- |\n| Bo | 5 |\n\n"
    "See [the docs](https://example.invalid/docs) or https://example.invalid/raw \U0001f600.\n\n"
    "```python\nprint('hi')\n```\n\n"
    "Inline `code` here. Fish &amp; chips&nbsp;now.\n"
)


# --- sanitize_spoken: individual transformations ---


def test_sanitize_empty() -> None:
    assert sanitize_spoken("") == ""


def test_sanitize_emphasis() -> None:
    text = "**bold** and *ital* and _under_ and __dunder__"
    assert sanitize_spoken(text) == "bold and ital and under and dunder"


def test_sanitize_inline_code() -> None:
    assert sanitize_spoken("run `ls -la` now") == "run ls -la now"


def test_sanitize_fenced_blocks_replaced_once() -> None:
    text = "before\n```py\nx = 1\n```\nafter\n```\nsecret_stuff\n```\nend"
    out = sanitize_spoken(text)
    assert out.count("skip the technical details") == 1
    assert "x = 1" not in out
    assert "secret_stuff" not in out
    assert out.startswith("before")
    assert out.endswith("end")


def test_sanitize_unterminated_fence_dropped() -> None:
    out = sanitize_spoken("intro\n```\nnever closed")
    assert out.startswith("intro")
    assert "never closed" not in out
    assert "skip the technical details" in out


def test_sanitize_headings() -> None:
    assert sanitize_spoken("## Big Title\nBody text") == "Big Title Body text"


def test_sanitize_bullets_become_sentences() -> None:
    assert sanitize_spoken("- alpha\n- beta three\n") == "alpha. beta three."


def test_sanitize_numbered_lists() -> None:
    assert sanitize_spoken("1. one\n2) two") == "one. two."


def test_sanitize_tables_flattened() -> None:
    text = "| Name | Age |\n| --- | --- |\n| Bo | 5 |"
    assert sanitize_spoken(text) == "Name, Age. Bo, 5."


def test_sanitize_links_keep_label() -> None:
    out = sanitize_spoken("Read [the guide](https://example.invalid/g) today")
    assert out == "Read the guide today"


def test_sanitize_bare_urls_replaced() -> None:
    out = sanitize_spoken("Go to https://example.invalid/page now")
    assert out == "Go to a link I can send you now"
    assert "http" not in out


def test_sanitize_emoji_removed() -> None:
    out = sanitize_spoken("Great job \U0001f389\U0001f600 team \u2b50")
    assert out == "Great job team"


def test_sanitize_html_entities() -> None:
    assert sanitize_spoken("fish &amp; chips&nbsp;and more") == "fish & chips and more"


def test_sanitize_whitespace_collapsed() -> None:
    assert sanitize_spoken("a\n\n  b\t\tc") == "a b c"


def test_sanitize_gnarly_document() -> None:
    out = sanitize_spoken(GNARLY)
    for piece in (
        "Report",
        "Hello world",
        "first point.",
        "one. two.",
        "Name, Age. Bo, 5.",
        "the docs",
        "a link I can send you",
        "skip the technical details",
        "Fish & chips now",
    ):
        assert piece in out
    assert not re.search(r"[*_#`|]", out)
    assert "http" not in out
    assert "\U0001f600" not in out


def test_sanitize_idempotent() -> None:
    once = sanitize_spoken(GNARLY)
    assert sanitize_spoken(once) == once
    plain = sanitize_spoken("Just a normal sentence, with 2 clauses.")
    assert sanitize_spoken(plain) == plain


def test_sanitize_adversarial_fence_input_is_fast() -> None:
    start = time.perf_counter()
    out = sanitize_spoken("```" + "a" * 20000)
    elapsed = time.perf_counter() - start
    assert elapsed < 0.2
    assert "skip the technical details" in out
    assert "a" * 100 not in out  # everything after the fence is code, dropped


def test_sanitize_adversarial_link_input_is_fast() -> None:
    start = time.perf_counter()
    out = sanitize_spoken("[](" * 5000)
    elapsed = time.perf_counter() - start
    assert elapsed < 0.2
    assert "[](" in out  # not a link, passed through (whitespace-collapsed)


def test_sanitize_truncates_oversized_input() -> None:
    out = sanitize_spoken("word " * 4000)  # 20000 chars of plain prose
    assert out.endswith("And there is more detail I can share if you want.")
    assert len(out) <= MAX_SPOKEN_INPUT


def test_sanitize_truncation_is_idempotent() -> None:
    once = sanitize_spoken("word " * 4000)
    assert sanitize_spoken(once) == once


def test_sanitize_input_at_cap_untouched() -> None:
    text = "a" * MAX_SPOKEN_INPUT
    assert sanitize_spoken(text) == text


# --- DeltaSanitizer ---


def test_delta_sanitizer_construct_split_across_three_chunks() -> None:
    chunks = ["Hello **wor", "ld** and `co", "de` done"]
    ds = DeltaSanitizer()
    outputs = [ds.feed(chunk) for chunk in chunks]
    outputs.append(ds.flush())
    total = "".join(outputs)
    assert total == sanitize_spoken("".join(chunks))
    assert outputs[0] == "Hello"  # emitted before the construct resolved


def test_delta_sanitizer_releases_oversized_buffer() -> None:
    ds = DeltaSanitizer()
    out = ds.feed("start ```\n" + "x" * 600)
    assert out.startswith("start")
    assert "skip the technical details" in out
    assert ds.flush() == ""


def test_delta_sanitizer_flush_returns_remainder() -> None:
    ds = DeltaSanitizer()
    first = ds.feed("tail **bo")
    assert first == "tail"
    total = first + ds.flush()
    assert total == sanitize_spoken("tail **bo")
    assert ds.flush() == ""


def test_delta_sanitizer_plain_prose_streams_through() -> None:
    ds = DeltaSanitizer()
    assert ds.feed("How are ") == "How are"
    assert ds.feed("you today?") == " you today?"
    assert ds.flush() == ""


# --- FillerPicker ---


def test_filler_picker_rejects_empty() -> None:
    with pytest.raises(ValueError, match="phrase"):
        FillerPicker([])


def test_filler_picker_single_phrase() -> None:
    picker = FillerPicker(["one moment"])
    assert picker.pick() == "one moment"
    assert picker.pick() == "one moment"


def test_filler_picker_never_repeats_consecutively() -> None:
    random.seed(1234)
    phrases = ["one", "two", "three", "four"]
    picker = FillerPicker(phrases)
    picks = [picker.pick() for _ in range(200)]
    assert all(a != b for a, b in zip(picks, picks[1:], strict=False))
    assert set(picks) <= set(phrases)


# --- build_opening ---


def test_build_opening_inbound_uses_greeting() -> None:
    settings = make_settings(greeting="Hello there, how can I help?")
    assert build_opening(settings, direction="inbound") == "Hello there, how can I help?"


def test_build_opening_outbound_with_purpose() -> None:
    settings = make_settings(assistant_name="Nova", principal="Mike")
    out = build_opening(
        settings,
        direction="outbound",
        dynamic_variables={"purpose": "I'm calling about your order."},
    )
    assert "Nova" in out
    assert "Mike" in out
    assert out.endswith("I'm calling about your order.")


def test_build_opening_missing_purpose_is_clean() -> None:
    settings = make_settings(assistant_name="Nova", principal="Mike")
    out = build_opening(settings, direction="outbound")
    assert "{" not in out
    assert "}" not in out
    assert "  " not in out
    assert out == out.strip()
    assert out.endswith("recorded.")


# --- build_instructions / VOICE_SYSTEM_PROMPT ---


def test_build_instructions_no_tools_sentence() -> None:
    text = build_instructions(make_settings(), direction="inbound")
    assert "Do not use any tools on this call" in text
    assert "Answer only from what you already know" in text


def test_build_instructions_with_tool_policy() -> None:
    settings = make_settings(
        tool_policy=ToolPolicy(
            enabled_tools=["send_message", "search_web"],
            confirm_tools=["send_message"],
            max_tool_calls_per_turn=2,
            max_tool_seconds_per_call=30.0,
        )
    )
    text = build_instructions(settings, direction="inbound")
    assert "send_message" in text
    assert "search_web" in text
    assert "explicit yes" in text
    assert "irreversible" in text
    assert "2 tool calls" in text
    assert "30 seconds" in text


def test_build_instructions_identity_outbound() -> None:
    settings = make_settings(assistant_name="Nova", principal="Mike")
    text = build_instructions(settings, direction="outbound")
    assert "Nova" in text
    assert "Mike" in text
    assert "on behalf of" in text
    assert "recorded" in text


def test_build_instructions_identity_inbound() -> None:
    settings = make_settings(assistant_name="Nova", principal="Mike")
    text = build_instructions(settings, direction="inbound")
    assert "Nova" in text
    assert "on behalf of" not in text


def test_voice_system_prompt_shape() -> None:
    low = VOICE_SYSTEM_PROMPT.lower()
    assert "phone" in low
    assert "markdown" in low
    assert VOICE_SYSTEM_PROMPT.strip() in build_instructions(make_settings(), direction="inbound")
    assert len(VOICE_SYSTEM_PROMPT.strip().splitlines()) <= 40
