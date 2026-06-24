"""Integration tests for pagen.pipeline.

The worker functions are exercised directly and in-process so render_document
can be mocked (the Pool uses forkserver, which would not inherit a monkeypatch).
The full Pool-based generate_split is covered by a real-render slow test.
"""

import json
import os

import cv2
import numpy as np
import pytest

from pagen import pipeline
from pagen.pipeline import DatasetTask, EvalTask
from pagen.augment import AugmentContext


def _real_png(w=16, h=20):
    img = np.full((h, w, 3), 255, dtype=np.uint8)
    ok, enc = cv2.imencode(".png", img)
    assert ok
    return enc.tobytes()


def _fake_pages(*args, **kwargs):
    from pagen.render import Page
    return [Page(
        png_bytes=_real_png(),
        polygons=[[[0, 0], [1, 0], [1, 1], [0, 1]]],
        labels=["w"],
        plain_text="w",
        width=16,
        height=20,
    )]


def _template(tmp_path):
    tpl = tmp_path / "t.md"
    tpl.write_text("# {WORDS_1}", encoding="utf-8")
    return str(tpl)


def _dataset_task(tmp_path, **overrides):
    base = dict(
        output_dir=str(tmp_path / "train"),
        file_id=1,
        template_paths=[_template(tmp_path)],
        fonts=[],
        words=["a"],
        dpi=72,
        augment=False,
        augment_ctx=None,
        llm_config=None,
        keep_txt=False,
        keep_pdf=False,
        seed=1,
    )
    base.update(overrides)
    return DatasetTask(**base)


# ---------------------------------------------------------------------------
# _dataset_worker
# ---------------------------------------------------------------------------

def test_dataset_worker_no_augment_writes_image_and_entry(tmp_path, monkeypatch):
    monkeypatch.setattr("pagen.render.render_document", _fake_pages)
    task = _dataset_task(tmp_path, keep_txt=True)
    results = pipeline._dataset_worker(task)

    assert len(results) == 1
    name, entry = results[0]
    assert name == "1.png"
    assert os.path.exists(tmp_path / "train" / "images" / "1.png")
    assert os.path.exists(tmp_path / "train" / "1.txt")
    assert entry["labels"] == ["w"]
    assert entry["img_dimensions"] == [16, 20]
    assert len(entry["img_hash"]) == 64


def test_dataset_worker_augment_clean_path(tmp_path, monkeypatch):
    monkeypatch.setattr("pagen.render.render_document", _fake_pages)
    ctx = AugmentContext(clean_prob=1.0, scan_prob=0.0)
    task = _dataset_task(tmp_path, augment=True, augment_ctx=ctx)
    results = pipeline._dataset_worker(task)
    assert len(results) == 1
    assert os.path.exists(tmp_path / "train" / "images" / "1.png")


def test_dataset_worker_render_empty_returns_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr("pagen.render.render_document", lambda *a, **k: [])
    assert pipeline._dataset_worker(_dataset_task(tmp_path)) == []


# ---------------------------------------------------------------------------
# _eval_worker
# ---------------------------------------------------------------------------

def test_eval_worker_writes_png_and_text(tmp_path, monkeypatch):
    monkeypatch.setattr("pagen.render.render_document", _fake_pages)
    task = EvalTask(
        output_dir=str(tmp_path / "eval"),
        file_id=1,
        template_paths=[_template(tmp_path)],
        fonts=[],
        words=["a"],
        dpi=72,
        llm_config=None,
        seed=1,
    )
    result = pipeline._eval_worker(task)
    assert result == ["1"]
    assert os.path.exists(tmp_path / "eval" / "1.png")
    assert (tmp_path / "eval" / "1.txt").read_text(encoding="utf-8") == "w"


# ---------------------------------------------------------------------------
# Full Pool-based pipeline — real rendering (slow)
# ---------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.render
def test_generate_split_end_to_end_and_append(tmp_path):
    from pagen._paths import DEFAULT_FONT

    tpl = tmp_path / "t.md"
    tpl.write_text("# عنوان\n\n{WORDS_3}\n", encoding="utf-8")
    out = str(tmp_path / "train")
    common = dict(
        template_paths=[str(tpl)],
        fonts=[DEFAULT_FONT],
        words=["كلمة", "مثال", "نص"],
        dpi=72,
        augment=False,
        workers=1,
        seed=1,
    )

    pipeline.generate_split(out, count=2, **common)
    labels = json.loads((tmp_path / "train" / "labels.json").read_text(encoding="utf-8"))
    assert len(labels) >= 1
    for entry in labels.values():
        assert len(entry["polygons"]) == len(entry["labels"])

    # Append: a second run must keep existing entries and add new ids.
    pipeline.generate_split(out, count=1, **common)
    labels2 = json.loads((tmp_path / "train" / "labels.json").read_text(encoding="utf-8"))
    assert set(labels).issubset(set(labels2))
    assert len(labels2) > len(labels)


@pytest.mark.slow
@pytest.mark.render
def test_generate_eval_end_to_end(tmp_path):
    from pagen._paths import DEFAULT_FONT

    tpl = tmp_path / "t.md"
    tpl.write_text("# عنوان\n\n{WORDS_3}\n", encoding="utf-8")
    out = str(tmp_path / "eval")
    pipeline.generate_eval(
        out, count=2,
        template_paths=[str(tpl)], fonts=[DEFAULT_FONT],
        words=["كلمة", "مثال", "نص"], dpi=72, workers=1, seed=1,
    )
    pngs = [p for p in os.listdir(out) if p.endswith(".png")]
    txts = [p for p in os.listdir(out) if p.endswith(".txt")]
    assert len(pngs) == 2
    assert len(txts) == 2          # one plain-text ground-truth per page
    # Each image has a matching .txt ground-truth file.
    for png in pngs:
        assert png[:-4] + ".txt" in txts
