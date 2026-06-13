import os
import argparse
import random
import re
from datetime import datetime, timedelta
import markdown
from bs4 import BeautifulSoup

try:
    from weasyprint import HTML, CSS
    from weasyprint.text.fonts import FontConfiguration

    WEASYPRINT_AVAILABLE = True
except ImportError:
    WEASYPRINT_AVAILABLE = False
    print("WARNING: weasyprint is not installed. PDF generation will fail.")

try:
    from pdf2image import convert_from_path

    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False
    print("WARNING: pdf2image is not installed. PNG generation will fail.")

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
            template_str = _normalize_ollama_output(response['message']['content'].strip())
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


def generate_document_pair(output_dir, use_ollama=False, ollama_model="llama3"):
    templates = [f for f in os.listdir(TEMPLATES_DIR) if f.endswith(".md")]
    if not templates:
        print("No templates found in", TEMPLATES_DIR)
        return False

    chosen_template = random.choice(templates)

    with open(os.path.join(TEMPLATES_DIR, chosen_template), "r", encoding="utf-8") as f:
        template_content = f.read()

    # Filenames
    base_name = str(len([f for f in os.listdir(output_dir) if f.endswith('.txt')]) + 1)
    txt_path = os.path.join(output_dir, f"{base_name}.txt")
    pdf_path = os.path.join(output_dir, f"{base_name}.pdf")
    png_path = os.path.join(output_dir, f"{base_name}.png")

    max_tries = 3
    for attempt in range(max_tries):
        # Generate finalized MD string with Eastern Digits
        generated_md = process_template(
            template_content, use_ollama=use_ollama, ollama_model=ollama_model
        )

        if not WEASYPRINT_AVAILABLE:
            # Cannot check pages, just accept
            doc_render = None
            break

        html_content = markdown.markdown(generated_md, extensions=["tables"])
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

        full_html = f"""
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
        
        doc_render = HTML(string=full_html).render()
        if len(doc_render.pages) <= 1:
            break
        elif attempt == max_tries - 1:
            print(f"WARNING: Document exceeded 1 page after {max_tries} tries.")
        else:
            print("Document exceeded 1 page. Retrying...")

    # 1. Save Pure Ground Truth (TXT, no markdown) - WITHOUT the file name on the 2nd page
    pure_text = strip_markdown_to_plain_text(generated_md)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(pure_text)

    # 2. Render to PDF
    if WEASYPRINT_AVAILABLE and doc_render:
        filename_only_html = f"""
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
                .filename-page {{
                    direction: ltr; /* English filename */
                    text-align: center;
                    font-size: 10pt; /* make it small to save ink */
                    color: #555; /* less intense black footprint */
                    margin-top: 2cm;
                }}
            </style>
        </head>
        <body>
            {html_content}
            <div class="page-break"></div>
            <div class="filename-page">
                file: {base_name}.png<br>
                template: {chosen_template}
            </div>
        </body>
        </html>
        """
        
        HTML(string=filename_only_html).write_pdf(pdf_path)

        font_msg = os.path.basename(chosen_font) if chosen_font else "System"

        if PDF2IMAGE_AVAILABLE:
            images = convert_from_path(pdf_path)
            if images:
                images[0].save(png_path, "PNG")
            print(
                f"Generated: {txt_path}, {pdf_path}, and {png_path} (Font: {font_msg})"
            )
        else:
            print(
                f"Generated: {txt_path} and {pdf_path} (Font: {font_msg}) (Skipped PNG due to missing pdf2image)"
            )

    else:
        print(f"Generated: {txt_path} (Skipped PDF/PNG due to missing weasyprint)")

    return pdf_path, png_path, txt_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate Eastern Arabic ground truth TXT, PDF, and PNG from MD templates."
    )
    parser.add_argument("-o", "--output", default="output", help="Output directory")
    parser.add_argument(
        "-c", "--count", type=int, default=1, help="Number of files to generate"
    )
    parser.add_argument(
        "--ollama",
        action="store_true",
        help="Use Ollama to generate document content natively",
    )
    parser.add_argument(
        "--ollama-model", default="llama3", help="Ollama model to use (default: llama3)"
    )

    args = parser.parse_args()

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    for i in range(args.count):
        print(f"Generating document {i+1}/{args.count}...")
        generate_document_pair(
            args.output, use_ollama=args.ollama, ollama_model=args.ollama_model
        )
