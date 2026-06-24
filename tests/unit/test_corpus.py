"""Unit tests for pagen.corpus (script-agnostic word loading)."""

import random

from pagen import corpus


def test_load_words_from_file(tmp_path):
    f = tmp_path / "words.txt"
    f.write_text("one two\nthree\n", encoding="utf-8")
    assert corpus.load_words(str(f)) == ["one", "two", "three"]


def test_load_words_from_directory_sorted(tmp_path):
    (tmp_path / "b.txt").write_text("beta\n", encoding="utf-8")
    (tmp_path / "a.txt").write_text("alpha\n", encoding="utf-8")
    # _dir_files sorts filenames: a.txt before b.txt.
    assert corpus.load_words(str(tmp_path)) == ["alpha", "beta"]


def test_load_words_missing_path_uses_fallback():
    words = corpus.load_words("/no/such/path")
    assert words == list(corpus._FALLBACK_WORDS)


def test_load_words_empty_file_uses_fallback(tmp_path):
    f = tmp_path / "empty.txt"
    f.write_text("   \n", encoding="utf-8")
    assert corpus.load_words(str(f)) == list(corpus._FALLBACK_WORDS)


def test_random_words_count_and_empty():
    random.seed(0)
    out = corpus.random_words(["a", "b"], 3)
    assert len(out.split()) == 3
    assert corpus.random_words([], 3) == ""
