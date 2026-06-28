"""Real weasyprint + pymupdf rendering (gated behind the ``render`` marker)."""

import cv2
import numpy as np
import pytest

from pagen import render
from pagen._paths import DEFAULT_FONT

pytestmark = pytest.mark.render


def test_render_document_produces_consistent_page():
    template = "# عنوان\n\n{WORDS_3}\n\n- بند أول\n- بند ثان\n"
    pages = render.render_document(
        template, fonts=[DEFAULT_FONT], words=["كلمة", "مثال", "نص"], dpi=72,
    )
    assert len(pages) >= 1
    page = pages[0]
    # Core invariant: one polygon per word label.
    assert len(page.polygons) == len(page.labels)
    assert page.labels  # something was rendered
    assert page.width > 0 and page.height > 0
    # PNG bytes decode to an image of the reported size.
    arr = cv2.imdecode(np.frombuffer(page.png_bytes, np.uint8), cv2.IMREAD_COLOR)
    assert arr is not None
    assert arr.shape[1] == page.width and arr.shape[0] == page.height
    assert page.plain_text.strip()
    # Every polygon is a 4-point quad inside the page bounds.
    for poly in page.polygons:
        assert len(poly) == 4
        for x, y in poly:
            assert 0 <= x <= page.width and 0 <= y <= page.height


def test_render_document_wide_table_stays_within_page():
    """A wide, content-heavy table must wrap inside the page box instead of
    overflowing it.  Overflowed words get clamped to the page edge, producing
    degenerate (zero-area) polygons for text that the rasteriser cropped away;
    with fixed layout + word breaking every word lands on-page with real area.
    """
    long_token = "كلمةطويلةجدًاغيرقابلةللكسرنهائيًاإطلاقًا"
    header = "| " + " | ".join(f"عمود{i}" for i in range(6)) + " |"
    sep = "|" + "---|" * 6
    row = "| " + " | ".join(long_token for _ in range(6)) + " |"
    template = f"# جدول\n\n{header}\n{sep}\n" + "\n".join(row for _ in range(3)) + "\n"

    pages = render.render_document(template, fonts=[DEFAULT_FONT], words=["كلمة"], dpi=72)
    assert pages
    page = pages[0]
    assert len(page.polygons) == len(page.labels)
    for poly in page.polygons:
        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        assert 0 <= min(xs) and max(xs) <= page.width
        assert 0 <= min(ys) and max(ys) <= page.height
        # No clipped-to-edge degenerate boxes: every visible word has real area.
        assert max(xs) > min(xs) and max(ys) > min(ys)


def test_render_document_keep_pdf_attaches_pdf_bytes():
    pages = render.render_document(
        "# عنوان\n\n{WORDS_2}\n", fonts=[DEFAULT_FONT], words=["a", "b"],
        dpi=72, keep_pdf=True,
    )
    assert pages and pages[0].pdf_bytes
    assert pages[0].pdf_bytes[:4] == b"%PDF"
