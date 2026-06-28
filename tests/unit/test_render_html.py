"""Unit tests for pagen.render HTML/polygon helpers.

These don't need weasyprint/pymupdf (the real-render path is in
tests/integration/test_render_document.py behind the ``render`` marker).
Layout markers that are Arabic-specific go through ``arabic_profile``.
"""

import io

from PIL import Image

from pagen import render


# ---------------------------------------------------------------------------
# List markers + word wrapping
# ---------------------------------------------------------------------------

def test_get_eastern_arabic_numeral():
    assert render._get_eastern_arabic_numeral("12") == "١٢"


def test_inject_list_markers_unordered_and_ordered():
    html = "<ul><li>a</li></ul><ol><li>x</li><li>y</li></ol>"
    out = render.inject_list_markers(html)
    assert "• " in out          # bullet for ul
    assert "١. " in out         # first ol item, eastern numeral
    assert "٢. " in out


def test_inject_list_markers_respects_ol_start():
    out = render.inject_list_markers('<ol start="3"><li>x</li></ol>')
    assert "٣. " in out


def test_wrap_words_counts_spans_and_keeps_whitespace():
    out = render.wrap_words_in_html("<p>one two three</p>")
    assert out.count(f'class="{render.WORD_SPAN_CLASS}"') == 3


def test_wrap_words_skips_script_and_style():
    out = render.wrap_words_in_html("<style>body{color:red}</style><p>word</p>")
    # The word inside <p> is wrapped; CSS text inside <style> is not.
    assert out.count(f'class="{render.WORD_SPAN_CLASS}"') == 1


# ---------------------------------------------------------------------------
# build_full_html — Arabic layout markers (quarantined)
# ---------------------------------------------------------------------------

def test_build_full_html_contains_profile_markers(arabic_profile):
    html = render.build_full_html("<p>hi</p>", "", "CustomArabicFont")
    for marker in arabic_profile.html_markers:
        assert marker in html
    assert "CustomArabicFont" in html


def test_build_full_html_table_css_wraps_instead_of_overflowing():
    """Wide tables must wrap inside the page box.  Without fixed layout +
    word breaking, a content-heavy table overflows the page edge and the
    rasteriser silently crops the trailing (RTL: leftmost) columns.
    Script-agnostic: this is page-layout behaviour, not Arabic-specific.
    """
    html = render.build_full_html("<table><tr><td>x</td></tr></table>", "", "F")
    assert "table-layout: fixed" in html
    assert "overflow-wrap: break-word" in html or "word-break: break-word" in html


# ---------------------------------------------------------------------------
# build_word_polygons — ink-tightening on a synthetic page
# ---------------------------------------------------------------------------

def _png_with_black_box():
    img = Image.new("L", (100, 100), color=255)
    for y in range(20, 60):
        for x in range(20, 60):
            img.putpixel((x, y), 0)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_build_word_polygons_shrinks_to_ink():
    png = _png_with_black_box()
    word_rects = [(10, 10, 70, 70)]
    glyph_boxes = [(40, 40, (20, 20, 60, 60))]   # cx,cy inside the word rect
    polys = render.build_word_polygons(png, word_rects, glyph_boxes)
    assert len(polys) == 1
    poly = polys[0]
    assert len(poly) == 4
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    # Tightened roughly to the 20..60 ink box (with small pad), inside the rect.
    assert 10 <= min(xs) <= 25
    assert 55 <= max(xs) <= 70
    assert 10 <= min(ys) <= 25
    assert 55 <= max(ys) <= 70


def test_build_word_polygons_falls_back_to_word_rect_without_glyphs():
    png = _png_with_black_box()
    word_rects = [(10, 10, 70, 70)]
    polys = render.build_word_polygons(png, word_rects, glyph_boxes=[])
    assert len(polys) == 1
    assert len(polys[0]) == 4


# ---------------------------------------------------------------------------
# render_document early-exit (no weasyprint call — fast)
# ---------------------------------------------------------------------------

def test_render_document_skips_when_unfillable(monkeypatch):
    # fill_template keeps returning an unfilled placeholder -> after retries the
    # function bails out with [] before ever touching weasyprint.
    monkeypatch.setattr("pagen.text.fill_template", lambda *a, **k: "{WORDS_2} left")
    pages = render.render_document("{WORDS_2}", fonts=[], words=["a"], dpi=72)
    assert pages == []
