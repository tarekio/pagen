"""Document rendering: Markdown → HTML → PDF → PNG + word polygons.

The key function is ``render_document()``, which returns a list of ``Page``
dataclasses — one per rendered page — each carrying the rasterised PNG bytes,
word polygons, word labels, and plain-text ground truth.  Nothing is written to
disk; callers decide what to persist.
"""

from __future__ import annotations

import io
import re
import os
import random
import urllib.request
from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

import markdown
from bs4 import BeautifulSoup, NavigableString
from PIL import Image

if TYPE_CHECKING:
    from pagen.llm import LLMConfig

try:
    from weasyprint import HTML as WP_HTML
    WEASYPRINT_AVAILABLE = True
except ImportError:
    WEASYPRINT_AVAILABLE = False

try:
    import fitz
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False


WORD_SPAN_CLASS = "pw"


@dataclass
class Page:
    png_bytes: bytes
    polygons: list
    labels: list[str]
    plain_text: str
    width: int
    height: int
    pdf_bytes: Optional[bytes] = None  # set when caller requests keep_pdf


# ---------------------------------------------------------------------------
# HTML building
# ---------------------------------------------------------------------------

def _get_eastern_arabic_numeral(n: str) -> str:
    mapping = {str(i): chr(0x0660 + i) for i in range(10)}
    return ''.join(mapping.get(ch, ch) for ch in n)


def inject_list_markers(html_content: str) -> str:
    """Inject list markers as real DOM text nodes so wrap_words picks them up."""
    soup = BeautifulSoup(html_content, "html.parser")
    for ul in soup.find_all("ul"):
        for li in ul.find_all("li", recursive=False):
            li.insert(0, NavigableString("• "))
    for ol in soup.find_all("ol"):
        start = int(ol.get("start", 1))
        for i, li in enumerate(ol.find_all("li", recursive=False)):
            li.insert(0, NavigableString(_get_eastern_arabic_numeral(str(i + start)) + ". "))
    return str(soup)


def wrap_words_in_html(html_content: str) -> str:
    """Wrap every visible word in a <span class="pw"> for per-word box extraction."""
    soup = BeautifulSoup(html_content, "html.parser")
    for text_node in list(soup.find_all(string=True)):
        if text_node.parent.name in ("script", "style"):
            continue
        s = str(text_node)
        if not s.strip():
            continue
        new_nodes = []
        for part in re.split(r"(\s+)", s):
            if part == "":
                continue
            if part.strip() == "":
                new_nodes.append(NavigableString(part))
            else:
                span = soup.new_tag("span", attrs={"class": WORD_SPAN_CLASS})
                span.string = part
                new_nodes.append(span)
        text_node.replace_with(*new_nodes)
    return str(soup)


def build_full_html(html_content: str, font_face_css: str, font_family: str) -> str:
    return f"""
    <html dir="rtl" lang="ar">
    <head>
        <meta charset="utf-8">
        <style>
            {font_face_css}
            @page {{ size: A4; margin: 2cm; }}
            body {{
                font-family: '{font_family}', sans-serif;
                direction: rtl;
                text-align: right;
                font-size: 14pt;
                line-height: 1.6;
            }}
            ul, ol {{
                direction: rtl;
                text-align: right;
                padding-right: 1.5em;
                padding-left: 0;
                list-style: none;
            }}
            li {{
                direction: rtl;
                text-align: right;
            }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 1em; margin-bottom: 1em; }}
            th, td {{ border: 1px solid #000; padding: 0.5em; text-align: right; }}
            h1, h2 {{ text-align: center; }}
            .page-break {{ page-break-before: always; }}
        </style>
    </head>
    <body>
        {html_content}
    </body>
    </html>
    """


# ---------------------------------------------------------------------------
# Box / polygon extraction
# ---------------------------------------------------------------------------

def extract_word_boxes(page_box, scale: float):
    """Extract per-word (x0,y0,x1,y1) boxes and labels from WeasyPrint layout tree."""
    groups: dict = {}
    order: list = []

    def visit(box):
        for child in getattr(box, "children", []) or []:
            visit(child)
        text = getattr(box, "text", None)
        element = getattr(box, "element", None)
        if not text or element is None:
            return
        if box.element_tag != "span" or WORD_SPAN_CLASS not in (element.get("class") or ""):
            return
        key = id(element)
        if key not in groups:
            groups[key] = {"text": "", "boxes": []}
            order.append(key)
        groups[key]["text"] += text
        groups[key]["boxes"].append((box.position_x, box.position_y, box.width, box.height))

    visit(page_box)

    rects, labels = [], []
    for key in order:
        g = groups[key]
        word = g["text"].strip()
        if not word:
            continue
        x0 = min(b[0] for b in g["boxes"]) * scale
        y0 = min(b[1] for b in g["boxes"]) * scale
        x1 = max(b[0] + b[2] for b in g["boxes"]) * scale
        y1 = max(b[1] + b[3] for b in g["boxes"]) * scale
        rects.append((x0, y0, x1, y1))
        labels.append(word)
    return rects, labels


def extract_glyph_boxes(page, mat):
    """Tight per-glyph (cx, cy, (x0,y0,x1,y1)) boxes from PyMuPDF in raster px."""
    boxes = []
    for block in page.get_text("rawdict").get("blocks", []):
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                for ch in span.get("chars", []):
                    if not ch["c"].strip():
                        continue
                    r = fitz.Rect(ch["bbox"]) * mat
                    boxes.append(((r.x0 + r.x1) / 2, (r.y0 + r.y1) / 2, (r.x0, r.y0, r.x1, r.y1)))
    return boxes


def build_word_polygons(png_bytes: bytes, word_rects, glyph_boxes, thresh=240, pad=1):
    """Build tight 4-point polygon per word by unioning glyph boxes and shrinking to ink."""
    img = Image.open(io.BytesIO(png_bytes)).convert("L")
    W, H = img.size
    polygons = []
    for wx0, wy0, wx1, wy1 in word_rects:
        gx0 = gy0 = gx1 = gy1 = None
        for cx, cy, (rx0, ry0, rx1, ry1) in glyph_boxes:
            if wx0 <= cx <= wx1 and wy0 <= cy <= wy1:
                gx0 = rx0 if gx0 is None else min(gx0, rx0)
                gy0 = ry0 if gy0 is None else min(gy0, ry0)
                gx1 = rx1 if gx1 is None else max(gx1, rx1)
                gy1 = ry1 if gy1 is None else max(gy1, ry1)
        if gx0 is None:
            gx0, gy0, gx1, gy1 = wx0, wy0, wx1, wy1

        cx0, cy0 = max(0, int(gx0)), max(0, int(gy0))
        cx1, cy1 = min(W, int(round(gx1))), min(H, int(round(gy1)))
        nx0, ny0, nx1, ny1 = cx0, cy0, cx1, cy1
        if cx1 > cx0 and cy1 > cy0:
            bb = (
                img.crop((cx0, cy0, cx1, cy1))
                .point(lambda v: 255 if v < thresh else 0)
                .getbbox()
            )
            if bb:
                ix0, iy0, ix1, iy1 = bb
                nx0 = max(0, cx0 + ix0 - pad)
                ny0 = max(0, cy0 + iy0 - pad)
                nx1 = min(W, cx0 + ix1 + pad)
                ny1 = min(H, cy0 + iy1 + pad)
        polygons.append([[nx0, ny0], [nx1, ny0], [nx1, ny1], [nx0, ny1]])
    return polygons


# ---------------------------------------------------------------------------
# Top-level render
# ---------------------------------------------------------------------------

def render_document(
    template_content: str,
    fonts: list[str],
    words: list[str],
    dpi: int = 150,
    llm_config: "Optional[LLMConfig]" = None,
    keep_pdf: bool = False,
    max_tries: int = 3,
) -> list[Page]:
    """Render a filled template to one or more Pages.  Writes nothing to disk.

    Args:
        template_content: Raw markdown template with {WORDS_N} etc placeholders.
        fonts: List of available .ttf font paths (pre-filtered, no color fonts).
        words: Corpus word list for random placeholder filling.
        dpi: Rasterisation resolution.
        llm_config: When set, fills template via LLM instead of random words.
        keep_pdf: When True, attaches pdf_bytes to each Page.
        max_tries: Retries for unfilled placeholders.
    """
    if not WEASYPRINT_AVAILABLE or not PYMUPDF_AVAILABLE:
        raise RuntimeError("Both weasyprint and pymupdf are required for rendering.")

    from pagen.text import fill_template, has_unfilled_placeholders, md_to_plain

    generated_md = None
    for attempt in range(max_tries):
        generated_md = fill_template(template_content, words, llm_config)
        if not has_unfilled_placeholders(generated_md):
            break
        if attempt < max_tries - 1:
            print(f"  WARNING: unfilled placeholders on attempt {attempt + 1}, retrying...")
    if generated_md is None or has_unfilled_placeholders(generated_md):
        print(f"  WARNING: could not fill template after {max_tries} tries, skipping.")
        return []

    plain_text = md_to_plain(generated_md)

    html_content = markdown.markdown(generated_md, extensions=["tables"])
    html_content = inject_list_markers(html_content)
    html_content = wrap_words_in_html(html_content)

    chosen_font = random.choice(fonts) if fonts else None
    font_face_css, font_family = "", "sans-serif"
    if chosen_font:
        font_family = "CustomArabicFont"
        font_uri = urllib.request.pathname2url(os.path.abspath(chosen_font))
        font_face_css = f"""
        @font-face {{
            font-family: 'CustomArabicFont';
            src: url('file:{font_uri}');
        }}
        """

    full_html = build_full_html(html_content, font_face_css, font_family)
    doc_render = WP_HTML(string=full_html).render()
    pdf_bytes = doc_render.write_pdf()

    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pdf_doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    pages = []
    for page_idx, wp_page in enumerate(doc_render.pages):
        pdf_page = pdf_doc[page_idx]
        pix = pdf_page.get_pixmap(matrix=mat)
        png_bytes = pix.tobytes("png")

        glyph_boxes = extract_glyph_boxes(pdf_page, mat)
        word_rects, labels = extract_word_boxes(wp_page._page_box, dpi / 96.0)
        polygons = build_word_polygons(png_bytes, word_rects, glyph_boxes)

        pages.append(Page(
            png_bytes=png_bytes,
            polygons=polygons,
            labels=labels,
            plain_text=plain_text,
            width=pix.width,
            height=pix.height,
            pdf_bytes=pdf_bytes if keep_pdf else None,
        ))

    pdf_doc.close()
    return pages
