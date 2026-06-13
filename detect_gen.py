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


_OLLAMA_CHAR_MAP = str.maketrans({
    # Explicit removal
    '\u0640': None,  # ـ
    # Hamza normalization
    '\u0672': '\u0623',  # ٲ -> أ
    '\u0673': '\u0625',  # ٳ -> إ
    '\u0676': '\u0648',  # ٶ -> و
    '\u0678': '\u064a',  # ٸ -> ي
    # Ta Marbuta
    '\u0629': '\u0629',  # ة -> ة
    '\u06c3': '\u0629',  # ۃ -> ة
    # Presentation forms -> standard
    '\ufee2': '\u0645',  # ﻢ -> م
    '\ufee4': '\u0645',  # ﻤ -> م
    '\ufee3': '\u0645',  # ﻣ -> م
    '\ufee1': '\u0645',  # ﻡ -> م
    '\ufe8e': '\u0627',  # ﺎ -> ا
    '\ufe8d': '\u0627',  # ﺍ -> ا
    '\ufeeb': '\u0647',  # ﻫ -> ه
    '\ufeec': '\u0647',  # ﻬ -> ه
    '\ufeea': '\u0647',  # ﻪ -> ه
    '\ufee9': '\u0647',  # ﻩ -> ه
    '\ufeab': '\u0630',  # ﺫ -> ذ
    '\ufeac': '\u0630',  # ﺬ -> ذ
    '\ufe91': '\u0628',  # ﺑ -> ب
    '\ufe92': '\u0628',  # ﺒ -> ب
    '\ufe90': '\u0628',  # ﺐ -> ب
    '\ufe8f': '\u0628',  # ﺏ -> ب
    '\ufea4': '\u062d',  # ﺤ -> ح
    '\ufea3': '\u062d',  # ﺣ -> ح
    '\ufea2': '\u062d',  # ﺢ -> ح
    '\ufea8': '\u062e',  # ﺨ -> خ
    '\ufea7': '\u062e',  # ﺧ -> خ
    '\ufea6': '\u062e',  # ﺦ -> خ
    '\ufeae': '\u0631',  # ﺮ -> ر
    '\ufe97': '\u062a',  # ﺗ -> ت
    '\ufe98': '\u062a',  # ﺘ -> ت
    '\ufe95': '\u062a',  # ﺕ -> ت
    '\ufe96': '\u0629',  # ﺖ -> ة
    '\ufea9': '\u062f',  # ﺩ -> د
    '\ufeaa': '\u062f',  # ﺪ -> د
    '\ufedb': '\u0643',  # ﻛ -> ك
    '\ufedc': '\u0643',  # ﻜ -> ك
    '\ufeda': '\u0643',  # ﻚ -> ك
    '\ufed9': '\u0643',  # ﻙ -> ك
    '\ufef4': '\u064a',  # ﻴ -> ي
    '\ufef3': '\u064a',  # ﻳ -> ي
    '\ufef2': '\u064a',  # ﻲ -> ي
    '\ufef1': '\u064a',  # ﻱ -> ي
    '\ufee8': '\u0646',  # ﻨ -> ن
    '\ufee7': '\u0646',  # ﻧ -> ن
    '\ufee6': '\u0646',  # ﻦ -> ن
    '\ufee5': '\u0646',  # ﻥ -> ن
    '\ufede': '\u0644',  # ﻞ -> ل
    '\ufedf': '\u0644',  # ﻟ -> ل
    '\ufee0': '\u0644',  # ﻠ -> ل
    '\ufedd': '\u0644',  # ﻝ -> ل
    '\ufeb4': '\u0633',  # ﺴ -> س
    '\ufeb3': '\u0633',  # ﺳ -> س
    '\ufeb2': '\u0633',  # ﺲ -> س
    '\ufeb8': '\u0634',  # ﺸ -> ش
    '\ufeb7': '\u0634',  # ﺷ -> ش
    '\ufeb6': '\u0634',  # ﺶ -> ش
    '\ufecc': '\u0639',  # ﻌ -> ع
    '\ufecb': '\u0639',  # ﻋ -> ع
    '\ufeca': '\u0639',  # ﻊ -> ع
    '\ufed0': '\u063a',  # ﻐ -> غ
    '\ufecf': '\u063a',  # ﻏ -> غ
    '\ufece': '\u063a',  # ﻎ -> غ
    '\ufed4': '\u0641',  # ﻔ -> ف
    '\ufed3': '\u0641',  # ﻓ -> ف
    '\ufed2': '\u0641',  # ﻒ -> ف
    '\ufed8': '\u0642',  # ﻘ -> ق
    '\ufed7': '\u0642',  # ﻗ -> ق
    '\ufed6': '\u0642',  # ﻖ -> ق
    '\ufec0': '\u0636',  # ﻀ -> ض
    '\ufebf': '\u0636',  # ﺿ -> ض
    '\ufebe': '\u0636',  # ﺾ -> ض
    '\ufebc': '\u0635',  # ﺼ -> ص
    '\ufebb': '\u0635',  # ﺻ -> ص
    '\ufeba': '\u0635',  # ﺺ -> ص
    '\ufec4': '\u0637',  # ﻄ -> ط
    '\ufec3': '\u0637',  # ﻃ -> ط
    '\ufec2': '\u0637',  # ﻂ -> ط
    '\ufec8': '\u0638',  # ﻈ -> ظ
    '\ufec7': '\u0638',  # ﻇ -> ظ
    '\ufec6': '\u0638',  # ﻆ -> ظ
    '\ufeb0': '\u0632',  # ﺰ -> ز
    '\ufeaf': '\u0632',  # ﺯ -> ز
    '\ufe9e': '\u062c',  # ﺞ -> ج
    '\ufe9f': '\u062c',  # ﺟ -> ج
    '\ufea0': '\u062c',  # ﺠ -> ج
    # Western digits -> Eastern Arabic
    '0': '\u0660',  # 0 -> ٠
    '1': '\u0661',  # 1 -> ١
    '2': '\u0662',  # 2 -> ٢
    '3': '\u0663',  # 3 -> ٣
    '4': '\u0664',  # 4 -> ٤
    '5': '\u0665',  # 5 -> ٥
    '6': '\u0666',  # 6 -> ٦
    '7': '\u0667',  # 7 -> ٧
    '8': '\u0668',  # 8 -> ٨
    '9': '\u0669',  # 9 -> ٩
    # Persian/Urdu digits -> Eastern Arabic
    '\u06f0': '\u0660',  # ۰ -> ٠
    '\u06f1': '\u0661',  # ۱ -> ١
    '\u06f2': '\u0662',  # ۲ -> ٢
    '\u06f3': '\u0663',  # ۳ -> ٣
    '\u06f4': '\u0664',  # ۴ -> ٤
    '\u06f5': '\u0665',  # ۵ -> ٥
    '\u06f6': '\u0666',  # ۶ -> ٦
    '\u06f7': '\u0667',  # ۷ -> ٧
    '\u06f8': '\u0668',  # ۸ -> ٨
    '\u06f9': '\u0669',  # ۹ -> ٩
    # Variant/extended letters -> standard Arabic
    '\u0679': '\u062a',  # ٹ -> ت
    '\u067a': '\u062a',  # ٺ -> ت
    '\u067c': '\u062a',  # ټ -> ت
    '\u0689': '\u062f',  # ډ -> د
    '\u068a': '\u062f',  # ڊ -> د
    '\u0693': '\u0631',  # ړ -> ر
    '\u0694': '\u0631',  # ڔ -> ر
    '\u0695': '\u0631',  # ڕ -> ر
    '\u0699': '\u0632',  # ڙ -> ز
    '\u069c': '\u0634',  # ڜ -> ش
    '\u06a0': '\u063a',  # ڠ -> غ
    '\u06a7': '\u0642',  # ڧ -> ق
    '\u06a8': '\u0642',  # ڨ -> ق
    '\u06aa': '\u0643',  # ڪ -> ك
    '\u06ab': '\u0643',  # ګ -> ك
    '\u06ac': '\u0643',  # ڬ -> ك
    '\u06ad': '\u0643',  # ڭ -> ك
    '\u06b0': '\u0643',  # ڰ -> ك
    '\u06b5': '\u0644',  # ڵ -> ل
    '\u06b7': '\u0644',  # ڷ -> ل
    '\u06ba': '\u0646',  # ں -> ن
    '\u06bc': '\u0646',  # ڼ -> ن
    '\u06be': '\u0647',  # ھ -> ه
    '\u06c1': '\u0647',  # ہ -> ه
    '\u06d5': '\u0647',  # ە -> ه
    '\u06c6': '\u0648',  # ۆ -> و
    '\u06c7': '\u0648',  # ۇ -> و
    '\u06c8': '\u0648',  # ۈ -> و
    '\u06c9': '\u0648',  # ۉ -> و
    '\u06cb': '\u0648',  # ۋ -> و
    '\u06e5': '\u0648',  # ۥ -> و
    '\u06ce': '\u064a',  # ێ -> ي
    '\u06d0': '\u064a',  # ې -> ي
    '\u06d2': '\u064a',  # ے -> ي
    '\u06d3': '\u064a',  # ۓ -> ي
    '\u06e6': '\u064a',  # ۦ -> ي
    '\u0649': '\u064a',  # ى -> ي
})

_TASHKEEL_RE = re.compile('[\u0610-\u061a\u064b-\u065f]')


def _normalize_ollama_output(text):
    """Strip tashkeel and normalize Arabic variant/presentation-form characters."""
    return _TASHKEEL_RE.sub('', text.translate(_OLLAMA_CHAR_MAP))



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
                messages=[{"role": "user", "content": prompt}],
                think=False,
            )
            template_str = _filter_ollama_output(_normalize_ollama_output(response['message']['content'].strip()))
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


def _is_color_font(path):
    try:
        from fontTools.ttLib import TTFont
        tables = TTFont(path).keys()
        return "COLR" in tables or "SVG " in tables or "sbix" in tables
    except Exception:
        return False


def get_random_font():
    if not os.path.exists(FONTS_DIR):
        return None
    fonts = [
        f for f in os.listdir(FONTS_DIR)
        if f.endswith(".ttf") and not _is_color_font(os.path.join(FONTS_DIR, f))
    ]
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


WORD_SPAN_CLASS = "pw"


def inject_list_markers(html_content):
    """Inject list markers as real DOM text nodes so wrap_words_in_html() picks them up.

    CSS ::marker pseudo-elements are rendered in the PNG but have no DOM text node, so
    they would appear in the image without a bounding box. Injecting the marker as a
    NavigableString inside each <li> ensures it gets a <span class="pw"> and a polygon.
    build_full_html() must set list-style:none to suppress the now-duplicate CSS marker."""
    soup = BeautifulSoup(html_content, "html.parser")

    for ul in soup.find_all("ul"):
        for li in ul.find_all("li", recursive=False):
            li.insert(0, NavigableString("• "))

    for ol in soup.find_all("ol"):
        start = int(ol.get("start", 1))
        for i, li in enumerate(ol.find_all("li", recursive=False)):
            marker = get_eastern_arabic_numeral(str(i + start)) + ". "
            li.insert(0, NavigableString(marker))

    return str(soup)


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
    "؟؛«»—،%!#$&'()*+,-./:;<=>?@[\\]^_`{|}~×÷“”‘’…٪٫"
)
_PLACEHOLDER_RE = re.compile(
    r"\{(?:WORDS_\d+|INT_\d+_\d+|FLOAT_[\d.]+_[\d.]+|DATE)\}"
)


def _filter_ollama_output(text):
    """Strip characters not in the allowed set, keeping whitespace intact."""
    return ''.join(ch for ch in text if ch.isspace() or ch in _ALLOWED_CHARS)


def _is_valid_generated_text(text):
    return not _PLACEHOLDER_RE.search(text)


def generate_document_pair(
    output_dir, file_id, use_ollama=False, ollama_model="llama3", dpi=150, keep_pdf=False
):
    """Return a list of (img_name, entry) tuples — one per rendered page."""
    templates = [f for f in os.listdir(TEMPLATES_DIR) if f.endswith(".md")]
    if not templates:
        print("No templates found in", TEMPLATES_DIR)
        return []

    chosen_template = random.choice(templates)

    with open(os.path.join(TEMPLATES_DIR, chosen_template), "r", encoding="utf-8") as f:
        template_content = f.read()

    if not WEASYPRINT_AVAILABLE or not PYMUPDF_AVAILABLE:
        print("Cannot generate: weasyprint and pymupdf are both required.")
        return []

    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    base_name = str(file_id)
    pdf_path = os.path.join(output_dir, f"{base_name}.pdf")

    # Retry only for unfilled placeholders; multi-page is handled gracefully below.
    max_tries = 3
    generated_md = None
    for attempt in range(max_tries):
        generated_md = process_template(
            template_content, use_ollama=use_ollama, ollama_model=ollama_model
        )
        if not _is_valid_generated_text(generated_md):
            if attempt < max_tries - 1:
                print("Generated text has unfilled placeholders. Retrying...")
                print("Generated text was:", repr(generated_md))
                continue
            print(f"WARNING: Could not generate valid text after {max_tries} tries, skipping.")
            return []
        break

    html_content = markdown.markdown(generated_md, extensions=["tables"])
    html_content = inject_list_markers(html_content)
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

    # PDF (in-memory; only persisted when --keep-pdf)
    pdf_bytes = doc_render.write_pdf()
    if keep_pdf:
        with open(pdf_path, "wb") as f:
            f.write(pdf_bytes)

    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pdf_doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    n_pages = len(doc_render.pages)
    font_msg = os.path.basename(chosen_font) if chosen_font else "System"
    results = []

    for page_idx, wp_page in enumerate(doc_render.pages):
        # Single-page docs keep the bare name; multi-page get a _p1/_p2/… suffix.
        page_suffix = f"_p{page_idx + 1}" if n_pages > 1 else ""
        page_base = f"{base_name}{page_suffix}"
        png_path = os.path.join(images_dir, f"{page_base}.png")
        txt_path = os.path.join(output_dir, f"{page_base}.txt")

        pdf_page = pdf_doc[page_idx]
        pix = pdf_page.get_pixmap(matrix=mat)
        png_bytes = pix.tobytes("png")
        with open(png_path, "wb") as f:
            f.write(png_bytes)

        glyph_boxes = extract_glyph_boxes(pdf_page, mat)
        # CSS px -> raster px: WeasyPrint uses 96 px/inch, raster is at `dpi`.
        word_rects, labels = extract_word_boxes(wp_page._page_box, dpi / 96.0)
        polygons = build_word_polygons(png_bytes, word_rects, glyph_boxes)

        # Ground truth: words from this page's layout tree, one per line.
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(labels))

        entry = {
            "img_dimensions": [pix.width, pix.height],
            "img_hash": hashlib.sha256(png_bytes).hexdigest(),
            "polygons": polygons,
            "labels": labels,
        }

        pdf_msg = f", {pdf_path}" if keep_pdf and page_idx == 0 else ""
        print(
            f"Generated: {png_path}, {txt_path}{pdf_msg} "
            f"({len(polygons)} polygons, Font: {font_msg})"
        )
        results.append((f"{page_base}.png", entry))

    pdf_doc.close()
    return results


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
        return []


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
    with open(labels_path, "w", encoding="utf-8") as out:
        out.write("{\n")
        first = [True]

        def _emit(key, value):
            sep = "" if first[0] else ",\n"
            out.write(sep + json.dumps(key) + ": " + json.dumps(value, ensure_ascii=False))
            first[0] = False

        for img_name, entry in existing_entries.items():
            _emit(img_name, entry)

        pool = multiprocessing.Pool(n_workers)
        try:
            for page_results in pool.imap_unordered(_worker, tasks):
                for img_name, entry in page_results:
                    _emit(img_name, entry)
                if page_results:
                    done += 1
                    print(f"  {done}/{args.count} done")
            pool.close()
            pool.join()
        except KeyboardInterrupt:
            print("\nInterrupted — terminating workers...")
            pool.terminate()
            pool.join()
        finally:
            out.write("\n}\n")

    total = len(existing_entries) + done
    print(f"Wrote {total} entries to {labels_path}")
