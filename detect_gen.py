import multiprocessing
import os
import argparse
import hashlib
import io
import json
import random
import re
from datetime import datetime, timedelta
import markdown
from bs4 import BeautifulSoup, NavigableString
from PIL import Image

try:
    from weasyprint import HTML, CSS
    from weasyprint.text.fonts import FontConfiguration

    WEASYPRINT_AVAILABLE = True
except ImportError:
    WEASYPRINT_AVAILABLE = False
    print("WARNING: weasyprint is not installed. PDF generation will fail.")

try:
    import fitz  # PyMuPDF

    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False
    print("WARNING: pymupdf is not installed. PNG and polygon generation will fail.")

WORDS_FILE = "corpus.txt"
TEMPLATES_DIR = "templates"
FONTS_DIR = "fonts"


def load_words():
    if not os.path.exists(WORDS_FILE):
        return ["كلمة", "مثال", "اختبار", "عربي", "بدون", "علامات"]
    with open(WORDS_FILE, "r", encoding="utf-8") as f:
        content = f.read().split()
        return [w.strip() for w in content if w.strip()]


WORDS_LIST = load_words()


def get_random_words(count):
    if not WORDS_LIST:
        return ""
    return " ".join(random.choices(WORDS_LIST, k=int(count)))


def get_eastern_arabic_numeral(number_str):
    mapping = {
        "0": "٠",
        "1": "١",
        "2": "٢",
        "3": "٣",
        "4": "٤",
        "5": "٥",
        "6": "٦",
        "7": "٧",
        "8": "٨",
        "9": "٩",
    }
    return "".join([mapping.get(c, c) for c in str(number_str)])


def to_eastern_arabic_text(text):
    mapping = {
        "0": "٠",
        "1": "١",
        "2": "٢",
        "3": "٣",
        "4": "٤",
        "5": "٥",
        "6": "٦",
        "7": "٧",
        "8": "٨",
        "9": "٩",
    }
    return "".join([mapping.get(c, c) for c in text])


def get_random_int(min_val, max_val):
    val = random.randint(int(min_val), int(max_val))
    return str(val)


def get_random_float(min_val, max_val):
    val = int(
        random.uniform(float(min_val), float(max_val))
    )  # Cast to int to remove decimal points
    return str(val)


def get_random_date():
    start = datetime.now() - timedelta(days=365)
    random_days = random.randint(0, 365)
    dt = start + timedelta(days=random_days)
    day = dt.strftime("%d")
    month = dt.strftime("%m")
    year = dt.strftime("%Y")
    return f"{day} {month} {year}"


def process_template(template_str, use_ollama=False, ollama_model="llama3"):
    if use_ollama:
        try:
            import ollama

            prompt = f"""
            You are an expert Arabic document generator. I will give you a markdown template with placeholders like {{WORDS_N}}, {{INT_A_B}}, {{FLOAT_A_B}}, and {{DATE}}. 
            You must GENERATE the actual content and naturally replace these placeholders:
            - {{WORDS_N}}: Replace with N realistic, context-appropriate Arabic words.
            - {{INT_A_B}}: Replace with a random integer between A and B.
            - {{FLOAT_A_B}}: Replace with a random decimal between A and B.
            - {{DATE}}: Replace with a realistic date.

            DO NOT include any curly braces or placeholders like `WORDS` or `INT` in your output.
            Return ONLY the completed markdown document text, nothing else. No explanations.
            DO NOT include diacritics in the generated Arabic words. Use plain Arabic text.
            IMPORTANT: Do not exceed the length of the provided template. Keep the content brief so it fits on a single page.

            Template:
            {template_str}
            """
            response = ollama.chat(
                model=ollama_model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
            )
            template_str = response["message"]["content"].strip()
        except ImportError:
            print(
                "WARNING: ollama library not found. Falling back to random words. To use ollama, run: pip install ollama"
            )
            use_ollama = False
        except Exception as e:
            print(
                f"WARNING: Ollama generation failed ({e}). Falling back to random words."
            )
            use_ollama = False

    def replace_words(match):
        return get_random_words(match.group(1))

    def replace_int(match):
        return get_random_int(match.group(1), match.group(2))

    def replace_float(match):
        return get_random_float(match.group(1), match.group(2))

    text = re.sub(r"\{WORDS_(\d+)\}", replace_words, template_str)
    text = re.sub(r"\{INT_(\d+)_(\d+)\}", replace_int, text)
    text = re.sub(r"\{FLOAT_([\d\.]+)_([\d\.]+)\}", replace_float, text)
    text = text.replace("{DATE}", get_random_date())

    # Finally, convert ALL standard numbers to Eastern Arabic strictly everywhere in the merged doc
    text = to_eastern_arabic_text(text)

    return text


def strip_markdown_to_plain_text(md_text):
    # Remove sequences of underscores from ground truth
    md_text = re.sub(r'_{2,}', '', md_text)
    html = markdown.markdown(md_text, extensions=["tables"])
    soup = BeautifulSoup(html, "html.parser")
    # Ground truth text strictly matching what will be shown, preserving table rows somewhat
    return "\n".join(
        [
            line.strip()
            for line in soup.get_text(separator="\n").splitlines()
            if line.strip()
        ]
    )


def get_random_font():
    if not os.path.exists(FONTS_DIR):
        return None
    fonts = [f for f in os.listdir(FONTS_DIR) if f.endswith(".ttf")]
    if not fonts:
        return None
    return os.path.join(FONTS_DIR, random.choice(fonts))


def build_full_html(html_content, font_face_css, font_family):
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


WORD_SPAN_CLASS = "pw"


def wrap_words_in_html(html_content):
    """Wrap every visible word (whitespace-separated token) of the rendered HTML in a
    `<span class="pw">` so each word becomes an addressable inline box, without changing
    layout. Whitespace and existing tags/structure (tables, headings, lists) are preserved."""
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


def extract_word_boxes(page_box, scale):
    """Per-word `(x0, y0, x1, y1)` boxes plus the matching word string, read from
    WeasyPrint's layout tree. Text comes from the source DOM (always correct, unlike PDF
    text extraction); the box is the word's em-box scaled from CSS px to raster px. The
    em-box is loose, so it is only used as the region in which to gather the tight glyph
    geometry later. Parallel lists."""
    # Group text fragments by their source span element so a word that wraps across a
    # line is rejoined into a single label + box.
    groups = {}
    order = []

    def visit(box):
        for child in getattr(box, "children", []) or []:
            visit(child)
        text = getattr(box, "text", None)
        element = getattr(box, "element", None)
        if not text or element is None:
            return
        if box.element_tag != "span" or WORD_SPAN_CLASS not in (
            element.get("class") or ""
        ):
            return
        key = id(element)
        if key not in groups:
            groups[key] = {"text": "", "boxes": []}
            order.append(key)
        groups[key]["text"] += text
        groups[key]["boxes"].append(
            (box.position_x, box.position_y, box.width, box.height)
        )

    visit(page_box)

    rects = []
    labels = []
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
    """Tight per-glyph boxes from PyMuPDF, as `(cx, cy, (x0, y0, x1, y1))` in raster px
    (centre + box). Unlike PyMuPDF's text, the glyph *geometry* is reliable and fully
    encloses descenders and detached marks (dots), which the em-box and pixel scanning
    miss. Whitespace glyphs are skipped."""
    boxes = []
    for block in page.get_text("rawdict").get("blocks", []):
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                for ch in span.get("chars", []):
                    if not ch["c"].strip():
                        continue
                    r = fitz.Rect(ch["bbox"]) * mat
                    boxes.append(
                        ((r.x0 + r.x1) / 2, (r.y0 + r.y1) / 2, (r.x0, r.y0, r.x1, r.y1))
                    )
    return boxes


def build_word_polygons(png_bytes, word_rects, glyph_boxes, thresh=240, pad=1):
    """Build a tight 4-point polygon (TL, TR, BR, BL) per word. For each WeasyPrint word
    box we union the PyMuPDF glyph boxes whose centre falls inside it — that region
    provably contains all of the word's ink (dots, tails) and nothing from neighbours —
    then shrink to the actual rendered ink to drop the loose top. Falls back to the
    em-box when no glyph matches."""
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
        if gx0 is None:  # no glyph centre landed in the em-box: keep the em-box
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


_ALLOWED_CHARS = frozenset(
    "٠١٢٣٤٥٦٧٨٩ءآأؤإئابةتثجحخدذرزسشصضطظعغفقكلمنهوىي"
    "؟؛«»—،%!#$&'()*+,-./:;<=>?@[\\]^_`{|}~×÷“”‘’…"
)
_PLACEHOLDER_RE = re.compile(
    r"\{(?:WORDS_\d+|INT_\d+_\d+|FLOAT_[\d.]+_[\d.]+|DATE)\}"
)


def _is_valid_generated_text(text):
    if _PLACEHOLDER_RE.search(text):
        return False
    return all(ch.isspace() or ch in _ALLOWED_CHARS for ch in text)


def generate_document_pair(
    output_dir, file_id, use_ollama=False, ollama_model="llama3", dpi=150, keep_pdf=False
):
    templates = [f for f in os.listdir(TEMPLATES_DIR) if f.endswith(".md")]
    if not templates:
        print("No templates found in", TEMPLATES_DIR)
        return None

    chosen_template = random.choice(templates)

    with open(os.path.join(TEMPLATES_DIR, chosen_template), "r", encoding="utf-8") as f:
        template_content = f.read()

    if not WEASYPRINT_AVAILABLE or not PYMUPDF_AVAILABLE:
        print("Cannot generate: weasyprint and pymupdf are both required.")
        return None

    # doctr layout: images in an `images/` subdir, labels.json at the dataset root.
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    base_name = str(file_id)
    png_path = os.path.join(images_dir, f"{base_name}.png")
    txt_path = os.path.join(output_dir, f"{base_name}.txt")
    pdf_path = os.path.join(output_dir, f"{base_name}.pdf")

    max_tries = 3
    for attempt in range(max_tries):
        # Generate finalized MD string with Eastern Digits
        generated_md = process_template(
            template_content, use_ollama=use_ollama, ollama_model=ollama_model
        )

        if not _is_valid_generated_text(generated_md):
            if attempt < max_tries - 1:
                print("Generated text has invalid chars/placeholders. Retrying...")
                continue
            print(f"WARNING: Could not generate valid text after {max_tries} tries, skipping.")
            return None

        html_content = markdown.markdown(generated_md, extensions=["tables"])
        html_content = wrap_words_in_html(html_content)
        chosen_font = get_random_font()
        font_face_css = ""
        font_family = "sans-serif"

        if chosen_font:
            font_family = "CustomArabicFont"
            import urllib.request

            font_uri = urllib.request.pathname2url(os.path.abspath(chosen_font))
            font_face_css = f"""
            @font-face {{
                font-family: 'CustomArabicFont';
                src: url('file:{font_uri}');
            }}
            """

        full_html = build_full_html(html_content, font_face_css, font_family)

        doc_render = HTML(string=full_html).render()
        if len(doc_render.pages) <= 1:
            break
        elif attempt == max_tries - 1:
            print(f"WARNING: Document exceeded 1 page after {max_tries} tries.")
        else:
            print("Document exceeded 1 page. Retrying...")

    # 1. Ground truth (plain text, no markdown)
    pure_text = strip_markdown_to_plain_text(generated_md)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(pure_text)

    # 2. PDF (in-memory; only persisted when --keep-pdf)
    pdf_bytes = doc_render.write_pdf()
    if keep_pdf:
        with open(pdf_path, "wb") as f:
            f.write(pdf_bytes)

    # 3. Rasterize the PDF to PNG and read tight glyph geometry from the same PDF.
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pdf_doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pdf_page = pdf_doc[0]
    pix = pdf_page.get_pixmap(matrix=mat)
    png_bytes = pix.tobytes("png")
    with open(png_path, "wb") as f:
        f.write(png_bytes)
    glyph_boxes = extract_glyph_boxes(pdf_page, mat)
    pdf_doc.close()

    # 4. Words + labels from WeasyPrint's layout tree (correct Arabic text); the polygon
    # geometry comes from the PyMuPDF glyph boxes inside each word, tightened to the ink.
    # CSS px -> raster px: WeasyPrint uses 96 px/inch, raster is rendered at `dpi`.
    word_rects, labels = extract_word_boxes(doc_render.pages[0]._page_box, dpi / 96.0)
    polygons = build_word_polygons(png_bytes, word_rects, glyph_boxes)
    entry = {
        "img_dimensions": [pix.width, pix.height],
        "img_hash": hashlib.sha256(png_bytes).hexdigest(),
        "polygons": polygons,
        "labels": labels,
    }

    font_msg = os.path.basename(chosen_font) if chosen_font else "System"
    pdf_msg = f", {pdf_path}" if keep_pdf else ""
    print(
        f"Generated: {png_path}, {txt_path}{pdf_msg} "
        f"({len(polygons)} polygons, Font: {font_msg})"
    )

    return f"{base_name}.png", entry


def _worker(task):
    output_dir, file_id, use_ollama, ollama_model, dpi, keep_pdf = task
    random.seed()  # reseed per-worker — fork inherits parent RNG state, causing duplicates
    try:
        return generate_document_pair(
            output_dir, file_id,
            use_ollama=use_ollama, ollama_model=ollama_model,
            dpi=dpi, keep_pdf=keep_pdf,
        )
    except Exception as e:
        print(f"WARNING: doc {file_id} failed: {e}")
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate Eastern Arabic text-detection data (PNG + ground-truth TXT + "
        "doctr-format polygon coord.json) from Markdown templates."
    )
    parser.add_argument("-o", "--output", default="output", help="Output directory")
    parser.add_argument(
        "-c", "--count", type=int, default=1, help="Number of files to generate"
    )
    parser.add_argument(
        "--dpi", type=int, default=150, help="Raster resolution for PNGs (default: 150)"
    )
    parser.add_argument(
        "--keep-pdf",
        action="store_true",
        help="Also save the intermediate PDF for each document",
    )
    parser.add_argument(
        "--ollama",
        action="store_true",
        help="Use Ollama to generate document content natively",
    )
    parser.add_argument(
        "--ollama-model", default="llama3", help="Ollama model to use (default: llama3)"
    )
    parser.add_argument(
        "--workers", type=int, default=os.cpu_count(),
        help="Number of parallel worker processes (default: all CPUs)",
    )

    args = parser.parse_args()

    images_dir = os.path.join(args.output, "images")
    os.makedirs(images_dir, exist_ok=True)

    # Pre-assign contiguous IDs in the parent before workers start — no runtime race.
    existing = [
        int(m.group(1))
        for f in os.listdir(images_dir)
        if (m := re.match(r"(\d+)\.png$", f))
    ]
    start_id = max(existing, default=0) + 1
    file_ids = list(range(start_id, start_id + args.count))

    tasks = [
        (args.output, fid, args.ollama, args.ollama_model, args.dpi, args.keep_pdf)
        for fid in file_ids
    ]
    n_workers = max(1, min(args.workers, args.count))

    labels_path = os.path.join(args.output, "labels.json")

    # Load existing entries once up-front (only populated on re-runs).
    existing_entries = {}
    if os.path.exists(labels_path):
        with open(labels_path, "r", encoding="utf-8") as f:
            existing_entries = json.load(f)

    # Stream each result directly to disk as workers finish — the parent never
    # accumulates the full new batch in memory.
    done = 0
    with open(labels_path, "w", encoding="utf-8") as out, \
         multiprocessing.Pool(n_workers) as pool:
        out.write("{\n")
        first = [True]

        def _emit(key, value):
            sep = "" if first[0] else ",\n"
            out.write(sep + json.dumps(key) + ": " + json.dumps(value, ensure_ascii=False))
            first[0] = False

        for img_name, entry in existing_entries.items():
            _emit(img_name, entry)

        for result in pool.imap_unordered(_worker, tasks):
            if result is None:
                continue
            img_name, entry = result
            _emit(img_name, entry)
            done += 1
            print(f"  {done}/{args.count} done")

        out.write("\n}\n")

    total = len(existing_entries) + done
    print(f"Wrote {total} entries to {labels_path}")
