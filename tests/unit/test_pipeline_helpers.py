"""Unit tests for pagen.pipeline ID assignment and incremental JSON writer.

All script-agnostic.  Heavier generate_split/generate_eval flows live in the
integration tests.
"""

import json

from pagen import pipeline


# ---------------------------------------------------------------------------
# _next_id / _next_id_plain
# ---------------------------------------------------------------------------

def test_next_id_missing_dir():
    assert pipeline._next_id("/no/such/images") == 1


def test_next_id_from_existing_pngs(tmp_path):
    (tmp_path / "1.png").write_bytes(b"")
    (tmp_path / "2.png").write_bytes(b"")
    assert pipeline._next_id(str(tmp_path)) == 3


def test_next_id_handles_multipage_suffix(tmp_path):
    (tmp_path / "5_p1.png").write_bytes(b"")
    (tmp_path / "5_p2.png").write_bytes(b"")
    (tmp_path / "notes.txt").write_text("x", encoding="utf-8")  # ignored
    assert pipeline._next_id(str(tmp_path)) == 6


def test_next_id_plain_empty_dir(tmp_path):
    assert pipeline._next_id_plain(str(tmp_path)) == 1


def test_next_id_plain_from_existing(tmp_path):
    (tmp_path / "3.png").write_bytes(b"")
    assert pipeline._next_id_plain(str(tmp_path)) == 4


# ---------------------------------------------------------------------------
# _JsonWriter
# ---------------------------------------------------------------------------

def test_jsonwriter_merges_existing_and_new(tmp_path):
    path = str(tmp_path / "labels.json")
    writer = pipeline._JsonWriter(path, {"a.png": {"labels": ["x"]}})
    writer.write("b.png", {"labels": ["y"]})
    writer.close()

    data = json.loads((tmp_path / "labels.json").read_text(encoding="utf-8"))
    assert set(data) == {"a.png", "b.png"}
    assert data["b.png"]["labels"] == ["y"]


def test_jsonwriter_preserves_unicode_unescaped(tmp_path):
    path = str(tmp_path / "labels.json")
    writer = pipeline._JsonWriter(path, {})
    writer.write("p.png", {"labels": ["مرحبا"]})
    writer.close()
    raw = (tmp_path / "labels.json").read_text(encoding="utf-8")
    assert "مرحبا" in raw          # ensure_ascii=False
    assert "\\u" not in raw


def test_jsonwriter_empty_is_valid_json(tmp_path):
    path = str(tmp_path / "labels.json")
    writer = pipeline._JsonWriter(path, {})
    writer.close()
    assert json.loads((tmp_path / "labels.json").read_text(encoding="utf-8")) == {}
