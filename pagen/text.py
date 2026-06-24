"""Arabic text processing: normalization, digit conversion, placeholder filling,
and plain-text ground truth extraction from rendered Markdown."""

from __future__ import annotations

import re
import random
from datetime import datetime, timedelta
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from pagen.llm import LLMConfig

# ---------------------------------------------------------------------------
# Arabic character normalization (presentation forms → standard codepoints)
# ---------------------------------------------------------------------------

_CHAR_MAP = str.maketrans({
    'ـ': None,   # ـ tatweel — remove
    # Hamza normalization
    'ٲ': 'أ', 'ٳ': 'إ', 'ٶ': 'و', 'ٸ': 'ي',
    # Ta Marbuta
    'ۃ': 'ة',
    # Presentation forms → standard
    'ﻢ': 'م', 'ﻤ': 'م', 'ﻣ': 'م', 'ﻡ': 'م',
    'ﺎ': 'ا', 'ﺍ': 'ا',
    'ﻫ': 'ه', 'ﻬ': 'ه', 'ﻪ': 'ه', 'ﻩ': 'ه',
    'ﺫ': 'ذ', 'ﺬ': 'ذ',
    'ﺑ': 'ب', 'ﺒ': 'ب', 'ﺐ': 'ب', 'ﺏ': 'ب',
    'ﺤ': 'ح', 'ﺣ': 'ح', 'ﺢ': 'ح',
    'ﺨ': 'خ', 'ﺧ': 'خ', 'ﺦ': 'خ',
    'ﺮ': 'ر',
    'ﺗ': 'ت', 'ﺘ': 'ت', 'ﺕ': 'ت', 'ﺖ': 'ة',
    'ﺩ': 'د', 'ﺪ': 'د',
    'ﻛ': 'ك', 'ﻜ': 'ك', 'ﻚ': 'ك', 'ﻙ': 'ك',
    'ﻴ': 'ي', 'ﻳ': 'ي', 'ﻲ': 'ي', 'ﻱ': 'ي',
    'ﻨ': 'ن', 'ﻧ': 'ن', 'ﻦ': 'ن', 'ﻥ': 'ن',
    'ﻞ': 'ل', 'ﻟ': 'ل', 'ﻠ': 'ل', 'ﻝ': 'ل',
    'ﺴ': 'س', 'ﺳ': 'س', 'ﺲ': 'س',
    'ﺸ': 'ش', 'ﺷ': 'ش', 'ﺶ': 'ش',
    'ﻌ': 'ع', 'ﻋ': 'ع', 'ﻊ': 'ع',
    'ﻐ': 'غ', 'ﻏ': 'غ', 'ﻎ': 'غ',
    'ﻔ': 'ف', 'ﻓ': 'ف', 'ﻒ': 'ف',
    'ﻘ': 'ق', 'ﻗ': 'ق', 'ﻖ': 'ق',
    'ﻀ': 'ض', 'ﺿ': 'ض', 'ﺾ': 'ض',
    'ﺼ': 'ص', 'ﺻ': 'ص', 'ﺺ': 'ص',
    'ﻄ': 'ط', 'ﻃ': 'ط', 'ﻂ': 'ط',
    'ﻈ': 'ظ', 'ﻇ': 'ظ', 'ﻆ': 'ظ',
    'ﺰ': 'ز', 'ﺯ': 'ز',
    'ﺞ': 'ج', 'ﺟ': 'ج', 'ﺠ': 'ج',
    # Western → Eastern Arabic digits
    '0': '٠', '1': '١', '2': '٢', '3': '٣', '4': '٤',
    '5': '٥', '6': '٦', '7': '٧', '8': '٨', '9': '٩',
    # Persian/Urdu → Eastern Arabic digits
    '۰': '٠', '۱': '١', '۲': '٢', '۳': '٣',
    '۴': '٤', '۵': '٥', '۶': '٦', '۷': '٧',
    '۸': '٨', '۹': '٩',
    # Variant/extended letters → standard Arabic
    'ٹ': 'ت', 'ٺ': 'ت', 'ټ': 'ت',
    'ډ': 'د', 'ڊ': 'د',
    'ړ': 'ر', 'ڔ': 'ر', 'ڕ': 'ر',
    'ڙ': 'ز', 'ڜ': 'ش', 'ڠ': 'غ',
    'ڧ': 'ق', 'ڨ': 'ق',
    'ڪ': 'ك', 'ګ': 'ك', 'ڬ': 'ك',
    'ڭ': 'ك', 'ڰ': 'ك',
    'ڵ': 'ل', 'ڷ': 'ل',
    'ں': 'ن', 'ڼ': 'ن',
    'ھ': 'ه', 'ہ': 'ه', 'ە': 'ه',
    'ۆ': 'و', 'ۇ': 'و', 'ۈ': 'و',
    'ۉ': 'و', 'ۋ': 'و', 'ۥ': 'و',
    'ێ': 'ي', 'ې': 'ي', 'ے': 'ي',
    'ۓ': 'ي', 'ۦ': 'ي', 'ى': 'ي',
})

_TASHKEEL_RE = re.compile(r'[ؐ-ًؚ-ٟ]')

_ALLOWED_CHARS = frozenset(
    "٠١٢٣٤٥٦٧٨٩ءآأؤإئابةتثجحخدذرزسشصضطظعغفقكلمنهوىي"
    "؟؛«»—،%!#$&'()*+,-./:;<=>?@[\\]^_`{|}~×÷“”‘’…٪٫"
)

_PLACEHOLDER_RE = re.compile(
    r"\{(?:WORDS_\d+|INT_\d+_\d+|FLOAT_[\d.]+_[\d.]+|DATE)\}"
)


def normalize(text: str) -> str:
    """Strip tashkeel and normalize Arabic variant/presentation-form characters."""
    return _TASHKEEL_RE.sub('', text.translate(_CHAR_MAP))


def filter_llm_output(text: str) -> str:
    """Strip characters outside the allowed Arabic/punctuation set (keep whitespace)."""
    return ''.join(ch for ch in text if ch.isspace() or ch in _ALLOWED_CHARS)


def to_eastern_digits(text: str) -> str:
    """Convert all Western ASCII digits in ``text`` to Eastern Arabic numerals."""
    mapping = {str(i): chr(0x0660 + i) for i in range(10)}
    return ''.join(mapping.get(ch, ch) for ch in text)


def has_unfilled_placeholders(text: str) -> bool:
    return bool(_PLACEHOLDER_RE.search(text))


# ---------------------------------------------------------------------------
# Random value generators
# ---------------------------------------------------------------------------

def _random_words(words: list[str], count: int) -> str:
    if not words:
        return ""
    return " ".join(random.choices(words, k=int(count)))


def _random_int(min_val: str, max_val: str) -> str:
    return str(random.randint(int(min_val), int(max_val)))


def _random_float(min_val: str, max_val: str) -> str:
    return f"{random.uniform(float(min_val), float(max_val)):.2f}"


def _random_date() -> str:
    start = datetime.now() - timedelta(days=365)
    dt = start + timedelta(days=random.randint(0, 365))
    return f"{dt.strftime('%d')} {dt.strftime('%m')} {dt.strftime('%Y')}"


# ---------------------------------------------------------------------------
# Template filling
# ---------------------------------------------------------------------------

_LLM_PROMPT = """\
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
{template}
"""


def fill_template(
    template: str,
    words: list[str],
    llm_config: "Optional[LLMConfig]" = None,
    max_tries: int = 3,
) -> str:
    """Fill a markdown template, returning completed Arabic markdown.

    When ``llm_config`` is provided the LLM fills in contextually appropriate
    content; otherwise placeholders are replaced with random corpus words/values.
    In both cases the final text has Western digits converted to Eastern Arabic.
    """
    if llm_config is not None:
        from pagen.llm import chat
        for attempt in range(max_tries):
            try:
                raw = chat(
                    llm_config,
                    [{"role": "user", "content": _LLM_PROMPT.format(template=template)}],
                )
                filled = filter_llm_output(normalize(raw))
                if not has_unfilled_placeholders(filled):
                    return to_eastern_digits(filled)
                if attempt < max_tries - 1:
                    continue
            except Exception as e:
                print(f"WARNING: LLM fill failed ({e}), falling back to random words.")
                break
        # fall through to random fill on failure

    text = re.sub(r"\{WORDS_(\d+)\}", lambda m: _random_words(words, m.group(1)), template)
    text = re.sub(r"\{INT_(\d+)_(\d+)\}", lambda m: _random_int(m.group(1), m.group(2)), text)
    text = re.sub(r"\{FLOAT_([\d.]+)_([\d.]+)\}", lambda m: _random_float(m.group(1), m.group(2)), text)
    text = text.replace("{DATE}", _random_date())
    return to_eastern_digits(text)


# ---------------------------------------------------------------------------
# Ground truth plain text extraction
# ---------------------------------------------------------------------------

def md_to_plain(md_text: str) -> str:
    """Strip Markdown formatting and return plain text matching rendered output."""
    import markdown
    from bs4 import BeautifulSoup

    md_text = re.sub(r'_{2,}', '', md_text)
    html = markdown.markdown(md_text, extensions=["tables"])
    soup = BeautifulSoup(html, "html.parser")
    return "\n".join(
        line.strip()
        for line in soup.get_text(separator="\n").splitlines()
        if line.strip()
    )
