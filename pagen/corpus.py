"""Word corpus loading and sampling for placeholder filling."""

import os
import random

from pagen._paths import CORPORA_DIR as DEFAULT_CORPORA_DIR

_FALLBACK_WORDS = ["كلمة", "مثال", "اختبار", "عربي", "بدون", "علامات"]


def load_words(corpus=None):
    """Load the word list from a file or a directory of files.

    Resolution order when ``corpus`` is None:
      1. the ``resources/corpora/`` directory (all files concatenated), if it exists and is non-empty
      2. a small built-in fallback list

    A ``corpus`` argument may point at either a file or a directory.
    """
    paths = []
    if corpus is None:
        if os.path.isdir(DEFAULT_CORPORA_DIR):
            paths = _dir_files(DEFAULT_CORPORA_DIR)
    elif os.path.isdir(corpus):
        paths = _dir_files(corpus)
    elif os.path.exists(corpus):
        paths = [corpus]

    words = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            words.extend(w.strip() for w in f.read().split() if w.strip())

    return words or list(_FALLBACK_WORDS)


def _dir_files(dirpath):
    return [
        os.path.join(dirpath, f)
        for f in sorted(os.listdir(dirpath))
        if os.path.isfile(os.path.join(dirpath, f))
    ]


def random_words(words, count):
    """Return ``count`` space-joined words sampled with replacement from ``words``."""
    if not words:
        return ""
    return " ".join(random.choices(words, k=int(count)))
