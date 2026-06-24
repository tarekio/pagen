"""Unit tests for pagen.text.

Most of this file is script-agnostic placeholder mechanics.  The few
Arabic-specific transforms go through the ``arabic_profile`` fixture so they
move cleanly when the tool becomes multi-script.
"""

import random

import pytest

from pagen import text


# ---------------------------------------------------------------------------
# Arabic-profile transforms (quarantined via fixture)
# ---------------------------------------------------------------------------

def test_normalize_presentation_and_variant_forms(arabic_profile):
    for raw, expected in arabic_profile.normalization_cases:
        assert text.normalize(raw) == expected


def test_normalize_strips_tashkeel(arabic_profile):
    # NOTE: stripping tashkeel is the *current default* only.  Post-experiments
    # the tool will optionally KEEP diacritics; when that lands, invert this one
    # test.  No other test may depend on tashkeel being removed.
    assert text.normalize(arabic_profile.tashkeel_word) == arabic_profile.tashkeel_stripped


def test_to_eastern_digits(arabic_profile):
    for western, eastern in arabic_profile.digit_pairs:
        assert text.to_eastern_digits(western) == eastern
    assert text.to_eastern_digits("abc") == "abc"


def test_filter_llm_output_drops_disallowed_keeps_space(arabic_profile):
    # Latin letters are not in the allowed set; whitespace survives.
    out = text.filter_llm_output(arabic_profile.disallowed_sample)
    assert out.strip() == ""
    assert " " in out
    # Allowed Arabic + punctuation passes through unchanged.
    assert text.filter_llm_output(arabic_profile.allowed_sample) == arabic_profile.allowed_sample


# ---------------------------------------------------------------------------
# Placeholder detection (script-agnostic)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("s", ["{WORDS_3}", "{INT_1_9}", "{FLOAT_1.5_9.0}", "{DATE}"])
def test_has_unfilled_placeholders_positive(s):
    assert text.has_unfilled_placeholders(s)


@pytest.mark.parametrize("s", ["", "plain text", "{UNKNOWN}", "{words_3}"])
def test_has_unfilled_placeholders_negative(s):
    assert not text.has_unfilled_placeholders(s)


# ---------------------------------------------------------------------------
# Random value generators (seeded for determinism)
# ---------------------------------------------------------------------------

def test_random_int_within_range():
    rng_seeded = random.Random(0)
    random.seed(0)
    for _ in range(50):
        v = int(text._random_int("5", "10"))
        assert 5 <= v <= 10


def test_random_float_within_range_is_decimal():
    # EXPECTS THE FIX (to be shipped separately on main): _random_float must
    # emit a real decimal, not a truncated integer.  This intentionally FAILS
    # against the current int()-truncating implementation in pagen/text.py,
    # standing as the spec for the pending fix.
    random.seed(1)
    for _ in range(50):
        s = text._random_float("2.0", "8.0")
        assert 2 <= float(s) <= 8
        assert "." in s   # a real decimal, not an integer string


def test_random_words_count_and_empty():
    random.seed(2)
    out = text._random_words(["a", "b", "c"], 4)
    assert len(out.split()) == 4
    assert text._random_words([], 4) == ""


def test_random_date_shape():
    random.seed(3)
    parts = text._random_date().split()
    assert len(parts) == 3
    assert all(p.isdigit() for p in parts)


# ---------------------------------------------------------------------------
# fill_template (non-LLM path) — script-agnostic mechanics
# ---------------------------------------------------------------------------

def test_fill_template_replaces_all_placeholders():
    random.seed(4)
    template = "{WORDS_2} {INT_1_5} {FLOAT_1_3} {DATE}"
    out = text.fill_template(template, words=["x", "y"], llm_config=None)
    assert not text.has_unfilled_placeholders(out)


def test_fill_template_converts_digits_to_eastern():
    random.seed(5)
    out = text.fill_template("{INT_100_100}", words=["x"], llm_config=None)
    # 100 -> eastern digits, no ASCII digits remain.
    assert not any(c.isdigit() and c.isascii() for c in out)


def test_fill_template_empty_words_leaves_no_placeholder():
    random.seed(6)
    out = text.fill_template("{WORDS_3}", words=[], llm_config=None)
    assert not text.has_unfilled_placeholders(out)


# ---------------------------------------------------------------------------
# fill_template (LLM path) — mocked, never hits network
# ---------------------------------------------------------------------------

def test_fill_template_llm_success(monkeypatch):
    # Use a normalize-stable word (no chars touched by _CHAR_MAP) so the
    # assertion checks LLM passthrough, not normalization.
    monkeypatch.setattr("pagen.llm.chat", lambda cfg, msgs: "محمد عربي ٥")
    out = text.fill_template("{WORDS_2}", words=["x"], llm_config=object())
    assert not text.has_unfilled_placeholders(out)
    assert "محمد" in out


def test_fill_template_llm_falls_back_on_exception(monkeypatch):
    def boom(cfg, msgs):
        raise RuntimeError("backend down")
    monkeypatch.setattr("pagen.llm.chat", boom)
    random.seed(7)
    out = text.fill_template("{WORDS_2}", words=["alpha", "beta"], llm_config=object())
    assert not text.has_unfilled_placeholders(out)


def test_fill_template_llm_falls_back_when_output_unfilled(monkeypatch):
    # LLM keeps returning a placeholder -> after retries, fall back to random.
    monkeypatch.setattr("pagen.llm.chat", lambda cfg, msgs: "{WORDS_2}")
    random.seed(8)
    out = text.fill_template("{WORDS_2}", words=["alpha", "beta"], llm_config=object())
    assert not text.has_unfilled_placeholders(out)


# ---------------------------------------------------------------------------
# md_to_plain (script-agnostic)
# ---------------------------------------------------------------------------

def test_md_to_plain_strips_markdown_and_blank_lines():
    md = "# Title\n\nSome **bold** text\n\n\n- item one\n- item two"
    plain = text.md_to_plain(md)
    lines = plain.splitlines()
    assert "Title" in plain
    assert "**" not in plain
    assert "" not in lines        # blank lines collapsed
    assert all(line == line.strip() for line in lines)


def test_md_to_plain_removes_underscore_runs():
    plain = text.md_to_plain("value: ____ end")
    assert "____" not in plain
