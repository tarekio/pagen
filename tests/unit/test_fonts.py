"""Unit tests for pagen.fonts (script-agnostic font discovery)."""

import random

from pagen import fonts


def test_list_fonts_missing_dir():
    assert fonts.list_fonts("/no/such/dir") == []


def test_list_fonts_filters_extension(tmp_path):
    # A fake .ttf (garbage bytes) is included: _is_color_font fails to parse it
    # and conservatively returns False (treat as usable).
    (tmp_path / "good.ttf").write_bytes(b"not a real font")
    (tmp_path / "notes.txt").write_text("ignore me", encoding="utf-8")
    found = fonts.list_fonts(str(tmp_path))
    assert [f.rsplit("/", 1)[-1] for f in found] == ["good.ttf"]


def test_is_color_font_handles_unparseable(tmp_path):
    p = tmp_path / "broken.ttf"
    p.write_bytes(b"garbage")
    assert fonts._is_color_font(str(p)) is False


def test_random_font_empty_pool_returns_none():
    assert fonts.random_font(fonts=[]) is None


def test_random_font_picks_from_pool():
    random.seed(0)
    pool = ["/a.ttf", "/b.ttf"]
    assert fonts.random_font(fonts=pool) in pool
