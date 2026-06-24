"""Unit tests for pagen.templates (LLM mocked; validation/slug/IO logic)."""

import random

import pytest

from pagen import templates


# ---------------------------------------------------------------------------
# _slugify
# ---------------------------------------------------------------------------

def test_slugify_ascii():
    assert templates._slugify("Invoice 2024") == "invoice_2024"


def test_slugify_pure_nonascii_token_uses_hash_fallback():
    # Single Arabic token with no surviving ascii -> deterministic hash slug.
    slug = templates._slugify("عقد")
    assert slug.startswith("doc_")
    assert templates._slugify("عقد") == slug   # deterministic


def test_slugify_strips_special_chars():
    assert templates._slugify("A/B: C!") == "ab_c"


# ---------------------------------------------------------------------------
# _validate
# ---------------------------------------------------------------------------

def test_validate_empty():
    ok, reason = templates._validate("   ")
    assert not ok and reason == "empty output"


def test_validate_no_placeholders():
    ok, reason = templates._validate("line one\nline two\nline three")
    assert not ok and reason == "no valid placeholders found"


def test_validate_invalid_brace():
    ok, reason = templates._validate("{WORDS_3}\n{BAD}\nthird line here")
    assert not ok and "invalid placeholder" in reason


def test_validate_too_short():
    ok, reason = templates._validate("{WORDS_3}")
    assert not ok and reason == "template too short"


def test_validate_ok():
    ok, reason = templates._validate("# {WORDS_3}\nbody line\nfooter {DATE}")
    assert ok and reason == ""


# ---------------------------------------------------------------------------
# pick_random
# ---------------------------------------------------------------------------

def test_pick_random_excludes_existing(tmp_path):
    existing_slug = templates.RANDOM_POOL[0][1]
    (tmp_path / f"{existing_slug}.md").write_text("x", encoding="utf-8")
    random.seed(0)
    picks = templates.pick_random(5, str(tmp_path))
    assert all(slug != existing_slug for _, slug in picks)


def test_pick_random_clamps_to_available(tmp_path):
    random.seed(1)
    picks = templates.pick_random(len(templates.RANDOM_POOL) + 50, str(tmp_path))
    assert len(picks) == len(templates.RANDOM_POOL)


def test_pick_random_empty_when_all_exist(tmp_path):
    for _, slug in templates.RANDOM_POOL:
        (tmp_path / f"{slug}.md").write_text("x", encoding="utf-8")
    assert templates.pick_random(3, str(tmp_path)) == []


# ---------------------------------------------------------------------------
# save_template
# ---------------------------------------------------------------------------

def test_save_template_writes_file(tmp_path):
    path = templates.save_template("content", "mydoc", str(tmp_path))
    assert path.endswith("mydoc.md")
    assert (tmp_path / "mydoc.md").read_text(encoding="utf-8") == "content\n"


def test_save_template_collision_suffix(tmp_path):
    templates.save_template("a", "dup", str(tmp_path))
    second = templates.save_template("b", "dup", str(tmp_path))
    assert second.endswith("dup_2.md")


# ---------------------------------------------------------------------------
# generate_template (LLM mocked)
# ---------------------------------------------------------------------------

def test_generate_template_strips_code_fence(monkeypatch):
    fenced = "```markdown\n# {WORDS_3}\nbody line\nfooter {DATE}\n```"
    monkeypatch.setattr(templates, "chat", lambda cfg, msgs: fenced)
    out = templates.generate_template("doc", llm_config=object())
    assert "```" not in out
    assert "{WORDS_3}" in out


def test_generate_template_retries_then_raises(monkeypatch):
    monkeypatch.setattr(templates, "chat", lambda cfg, msgs: "no placeholders here")
    with pytest.raises(RuntimeError):
        templates.generate_template("doc", llm_config=object())
