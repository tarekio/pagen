"""Shared fixtures for the pagen test suite.

Design note — *universal* direction:
The project is Arabic-only today but is heading toward multi-script support.
To keep the future refactor low-churn, every script-specific expectation lives
in the ``arabic_profile`` fixture below.  Script-agnostic tests (pipeline,
render structure, CLI, augment geometry, template mechanics) must NOT reference
it.  When a real LanguageProfile object is extracted, this fixture becomes the
single swap point and the same tests parametrize across profiles.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Script-specific expectations (quarantined here on purpose)
# ---------------------------------------------------------------------------

@pytest.fixture
def arabic_profile():
    """Everything that is specific to the current Arabic behaviour.

    Today these are hard-coded constants in pagen.text / pagen.render.  When
    the tool generalises, this fixture is what gets parametrized per language.
    """
    return SimpleNamespace(
        lang="ar",
        direction="rtl",
        # (western digit, expected eastern Arabic digit)
        digit_pairs=[("0", "٠"), ("5", "٥"), ("9", "٩")],
        # (input with presentation form / variant, expected normalized output)
        normalization_cases=[
            ("ﻣﺤﻤﺪ", "محمد"),      # presentation forms -> standard
            ("كلمةـ", "كلمة"),       # tatweel removed
            ("ٲحمد", "أحمد"),        # hamza variant normalized
        ],
        # A short tashkeel-bearing word and its stripped form (default today).
        tashkeel_word="مُحَمَّد",
        tashkeel_stripped="محمد",
        # Characters filter_llm_output must drop vs keep.
        disallowed_sample="Hello World",   # latin -> dropped entirely (spaces kept)
        allowed_sample="مرحبا، ٥٪",
        # Markers build_full_html must contain for this profile.
        html_markers=['dir="rtl"', 'lang="ar"', "text-align: right"],
    )


# ---------------------------------------------------------------------------
# Synthetic image helpers (fast: tiny ndarrays, no real photos)
# ---------------------------------------------------------------------------

@pytest.fixture
def page_bgr():
    """A small mostly-white 'rendered page' with a black ink block."""
    img = np.full((120, 90, 3), 255, dtype=np.uint8)
    img[40:80, 20:70] = 0   # a dark word-ish region
    return img


@pytest.fixture
def make_scene_image():
    """Factory writing a synthetic JPEG into a directory; returns the path."""
    import cv2
    import os

    def _make(directory: str, name: str, w: int = 200, h: int = 260):
        img = np.full((h, w, 3), 230, dtype=np.uint8)
        # a brighter inset 'paper' rectangle so detection has something to find
        img[30:h - 30, 30:w - 30] = 250
        path = os.path.join(directory, name)
        cv2.imwrite(path, img)
        return path

    return _make


# ---------------------------------------------------------------------------
# Fake render.Page (lets pipeline tests skip weasyprint/pymupdf)
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_page():
    """Build a minimal valid render.Page with a real decodable PNG."""
    import cv2
    from pagen.render import Page

    def _make(width: int = 40, height: int = 30, n_words: int = 2):
        img = np.full((height, width, 3), 255, dtype=np.uint8)
        ok, enc = cv2.imencode(".png", img)
        assert ok
        polygons = [[[0, 0], [5, 0], [5, 5], [0, 5]] for _ in range(n_words)]
        labels = [f"w{i}" for i in range(n_words)]
        return Page(
            png_bytes=enc.tobytes(),
            polygons=polygons,
            labels=labels,
            plain_text="\n".join(labels),
            width=width,
            height=height,
        )

    return _make
