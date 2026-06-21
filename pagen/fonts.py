"""Arabic font discovery and selection."""

import os
import random

from pagen._paths import FONTS_DIR as DEFAULT_FONTS_DIR


def _is_color_font(path):
    """Color fonts (COLR/SVG/sbix) render as empty boxes in WeasyPrint — skip them."""
    try:
        from fontTools.ttLib import TTFont
        tables = TTFont(path).keys()
        return "COLR" in tables or "SVG " in tables or "sbix" in tables
    except Exception:
        return False


def list_fonts(fonts_dir=DEFAULT_FONTS_DIR):
    """Return absolute paths of usable (non-color) .ttf fonts in ``fonts_dir``."""
    if not os.path.isdir(fonts_dir):
        return []
    return [
        os.path.join(fonts_dir, f)
        for f in os.listdir(fonts_dir)
        if f.endswith(".ttf") and not _is_color_font(os.path.join(fonts_dir, f))
    ]


def random_font(fonts_dir=DEFAULT_FONTS_DIR, fonts=None):
    """Pick a random usable font path, or None if none are available.

    Pass a precomputed ``fonts`` list to avoid re-scanning (and re-probing color
    tables) for every document.
    """
    pool = fonts if fonts is not None else list_fonts(fonts_dir)
    return random.choice(pool) if pool else None
