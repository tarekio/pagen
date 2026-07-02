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

# The fill prompt is data, not code: kept in resources/prompts/ so the eval
# harness can sweep prompt variants against the exact artifact the runtime ships.
# It uses a literal {{template}} marker (filled via str.replace, NOT str.format)
# so the {WORDS_N} examples inside the prompt stay untouched. The marker name
# matches promptfoo's nunjucks {{template}} variable, so the same file works in
# both places.
_FILL_PROMPT_CACHE: "Optional[str]" = None

# Fraction of non-space chars filter_llm_output may strip before an attempt is
# treated as off-target (English/emoji bleed) rather than valid Arabic.
_MAX_STRIP_FRACTION = 0.3


def _fill_prompt() -> str:
    """Load and cache the fill prompt template from resources/prompts/."""
    global _FILL_PROMPT_CACHE
    if _FILL_PROMPT_CACHE is None:
        from pagen._paths import FILL_PROMPT
        with open(FILL_PROMPT, encoding="utf-8") as f:
            _FILL_PROMPT_CACHE = f.read()
    return _FILL_PROMPT_CACHE


def _too_much_stripped(before: str, after: str, threshold: float = _MAX_STRIP_FRACTION) -> bool:
    """True when filtering removed more than ``threshold`` of the non-space chars.

    Catches output that drifted off the allowed Arabic set (e.g. an English
    paragraph): silently filtering it would leave holes in the ground-truth
    labels, so the attempt is rejected instead.
    """
    before_n = sum(1 for c in before if not c.isspace())
    if before_n == 0:
        return True
    after_n = sum(1 for c in after if not c.isspace())
    return (before_n - after_n) / before_n > threshold


def _llm_fill_once(template: str, llm_config: "LLMConfig") -> "Optional[str]":
    """One LLM fill attempt.

    Returns completed Arabic markdown, or ``None`` when the output is invalid
    (leftover placeholders or too much stripped by the allowed-char filter).
    Backend errors propagate to the caller, which decides retry vs fallback.
    """
    from pagen.llm import chat
    raw = chat(
        llm_config,
        [{"role": "user", "content": _fill_prompt().replace("{{template}}", template)}],
    )
    normalized = normalize(raw)
    filtered = filter_llm_output(normalized)
    if _too_much_stripped(normalized, filtered):
        return None
    if has_unfilled_placeholders(filtered):
        return None
    return to_eastern_digits(filtered)


def fill_random(template: str, words: list[str]) -> str:
    """Replace placeholders with random corpus words / values (no LLM)."""
    text = re.sub(r"\{WORDS_(\d+)\}", lambda m: _random_words(words, m.group(1)), template)
    text = re.sub(r"\{INT_(\d+)_(\d+)\}", lambda m: _random_int(m.group(1), m.group(2)), text)
    text = re.sub(r"\{FLOAT_([\d.]+)_([\d.]+)\}", lambda m: _random_float(m.group(1), m.group(2)), text)
    text = text.replace("{DATE}", _random_date())
    return to_eastern_digits(text)


def fill_template(
    template: str,
    words: list[str],
    llm_config: "Optional[LLMConfig]" = None,
    max_tries: int = 3,
) -> str:
    """Fill a markdown template, returning completed Arabic markdown.

    When ``llm_config`` is provided the LLM fills in contextually appropriate
    content, falling back to random corpus words on repeated invalid output or a
    backend error; otherwise placeholders are replaced with random words/values.
    In both cases the final text has Western digits converted to Eastern Arabic.
    """
    if llm_config is not None:
        for _ in range(max_tries):
            try:
                filled = _llm_fill_once(template, llm_config)
            except Exception as e:
                print(f"WARNING: LLM fill failed ({e}), falling back to random words.")
                break
            if filled is not None:
                return filled
        # all attempts invalid (or backend error): fall through to random fill

    return fill_random(template, words)


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
