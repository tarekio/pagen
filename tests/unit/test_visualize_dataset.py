"""Tests for pagen.visualize.visualize_dataset (the render/validate loop).

Saves overlays to disk; never uses show=True (that needs a GUI/matplotlib).
"""

import json

from PIL import Image

from pagen import visualize


def _make_dataset(tmp_path, entries, images):
    """Write images/ + labels.json; ``images`` maps name -> (w, h) to create."""
    ds = tmp_path / "ds"
    (ds / "images").mkdir(parents=True)
    for name, (w, h) in images.items():
        Image.new("RGB", (w, h), "white").save(ds / "images" / name)
    (ds / "labels.json").write_text(json.dumps(entries, ensure_ascii=False), encoding="utf-8")
    return ds


def _entry(w, h, polys=None, labels=None):
    return {
        "img_dimensions": [w, h],
        "img_hash": "x",
        "polygons": polys if polys is not None else [[[0, 0], [5, 0], [5, 5], [0, 5]]],
        "labels": labels if labels is not None else ["كلمة"],
    }


def test_visualize_saves_overlays(tmp_path, capsys):
    ds = _make_dataset(tmp_path, {"1.png": _entry(20, 20)}, {"1.png": (20, 20)})
    out = tmp_path / "vis"
    visualize.visualize_dataset(str(ds), save_dir=str(out), show=False, seed=0)
    assert (out / "1.png").exists()
    assert "0 problem(s) found" in capsys.readouterr().out


def test_visualize_default_save_dir(tmp_path):
    ds = _make_dataset(tmp_path, {"1.png": _entry(20, 20)}, {"1.png": (20, 20)})
    # save_dir=None and show=False -> defaults to <path>/debug_vis
    visualize.visualize_dataset(str(ds), save_dir=None, show=False, seed=0)
    assert (ds / "debug_vis" / "1.png").exists()


def test_visualize_skips_missing_image(tmp_path, capsys):
    # labels reference 2.png which is not on disk.
    ds = _make_dataset(
        tmp_path,
        {"1.png": _entry(20, 20), "2.png": _entry(20, 20)},
        {"1.png": (20, 20)},
    )
    visualize.visualize_dataset(str(ds), save_dir=str(tmp_path / "vis"), show=False, seed=0)
    out = capsys.readouterr().out
    assert "skip" in out
    assert "Visualized 1 image(s)" in out


def test_visualize_reports_problems(tmp_path, capsys):
    # Declared dimensions disagree with the real 20x20 image -> one problem.
    ds = _make_dataset(tmp_path, {"1.png": _entry(99, 99)}, {"1.png": (20, 20)})
    visualize.visualize_dataset(str(ds), save_dir=str(tmp_path / "vis"), show=False, seed=0)
    out = capsys.readouterr().out
    assert "1 problem(s) found" in out


def test_visualize_respects_max_images(tmp_path, capsys):
    entries = {f"{i}.png": _entry(20, 20) for i in range(1, 4)}
    images = {f"{i}.png": (20, 20) for i in range(1, 4)}
    ds = _make_dataset(tmp_path, entries, images)
    visualize.visualize_dataset(str(ds), save_dir=str(tmp_path / "vis"),
                                show=False, seed=0, max_images=1)
    assert "Visualized 1 image(s)" in capsys.readouterr().out
