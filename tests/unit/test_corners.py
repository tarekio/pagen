"""Unit tests for pagen.corners (helpers, overlay export, editor non-GUI logic).

The interactive cv2 event loop (_CornerEditor.run / launch_editor) is GUI and
not covered here.  Everything else — pure helpers, export_overlays, build_cache,
and the editor's provenance/coordinate logic — is exercised directly.
"""

import os

import numpy as np
import pytest

from pagen import corners, augment


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

def test_list_images_sorted_and_filtered(tmp_path, make_scene_image):
    make_scene_image(str(tmp_path), "b.jpg")
    make_scene_image(str(tmp_path), "a.png")
    (tmp_path / "notes.txt").write_text("ignore", encoding="utf-8")
    assert corners._list_images(str(tmp_path)) == ["a.png", "b.jpg"]


def test_full_frame_quad():
    assert corners._full_frame_quad(10, 20) == [[0, 0], [10, 0], [10, 20], [0, 20]]


def test_canonicalize_orders_and_clamps():
    # Out-of-order, out-of-bounds points -> TL,TR,BR,BL clamped to the frame.
    quad = [[-5, -5], [100, 0], [100, 100], [0, 100]]
    canon = corners._canonicalize(quad, 50, 50)
    assert canon == [[0, 0], [50, 0], [50, 50], [0, 50]]


def test_draw_quad_returns_image(page_bgr):
    quad = [[1, 1], [5, 1], [5, 5], [1, 5]]
    out = corners._draw_quad(page_bgr.copy(), quad)
    assert isinstance(out, np.ndarray)
    assert out.shape == page_bgr.shape


# ---------------------------------------------------------------------------
# build_cache
# ---------------------------------------------------------------------------

def test_build_cache_delegates_to_ensure(tmp_path, make_scene_image, monkeypatch):
    make_scene_image(str(tmp_path), "img.jpg")
    monkeypatch.setattr(augment, "detect_paper_quad",
                        lambda im: [[1, 1], [9, 1], [9, 9], [1, 9]])
    cache = corners.build_cache(str(tmp_path))
    assert "img.jpg" in cache


# ---------------------------------------------------------------------------
# export_overlays
# ---------------------------------------------------------------------------

def test_export_overlays_missing_cache_does_not_crash(tmp_path, capsys):
    out = tmp_path / "vis"
    corners.export_overlays(str(tmp_path), str(out))
    assert "not found" in capsys.readouterr().out
    assert not out.exists()


def test_export_overlays_writes_jpg(tmp_path, make_scene_image):
    make_scene_image(str(tmp_path), "img.jpg")
    augment.write_corners_cache(
        str(tmp_path),
        {"img.jpg": augment._make_entry([[5, 5], [50, 5], [50, 60], [5, 60]], "auto")},
    )
    out = tmp_path / "vis"
    corners.export_overlays(str(tmp_path), str(out))
    assert (out / "img.jpg").exists()


def test_export_overlays_skips_unreadable_image(tmp_path, capsys):
    # Cache references a file that isn't on disk -> [skip], no crash.
    augment.write_corners_cache(
        str(tmp_path),
        {"ghost.jpg": augment._make_entry([[0, 0], [1, 0], [1, 1], [0, 1]], "auto")},
    )
    out = tmp_path / "vis"
    corners.export_overlays(str(tmp_path), str(out))
    assert "skip" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# _CornerEditor — construction + non-GUI logic
# ---------------------------------------------------------------------------

def test_editor_init_empty_dir_raises(tmp_path):
    with pytest.raises(SystemExit):
        corners._CornerEditor(str(tmp_path), max_dim=1100)


def _editor_with_image(tmp_path, make_scene_image, monkeypatch, cache=None):
    make_scene_image(str(tmp_path), "img.jpg", w=200, h=260)
    if cache is not None:
        augment.write_corners_cache(str(tmp_path), cache)
    monkeypatch.setattr(corners, "detect_paper_quad",
                        lambda im: [[10, 10], [190, 10], [190, 250], [10, 250]])
    return corners._CornerEditor(str(tmp_path), max_dim=1100)


def test_editor_commit_keeps_source_when_unchanged(tmp_path, make_scene_image, monkeypatch):
    entry = augment._make_entry([[10, 10], [190, 10], [190, 250], [10, 250]], "auto")
    ed = _editor_with_image(tmp_path, make_scene_image, monkeypatch, cache={"img.jpg": entry})
    ed._commit()
    # Navigating past an unchanged image must NOT reclassify it as user-edited.
    assert ed.cache["img.jpg"]["source"] == "auto"


def test_editor_commit_marks_user_when_edited(tmp_path, make_scene_image, monkeypatch):
    entry = augment._make_entry([[10, 10], [190, 10], [190, 250], [10, 250]], "auto")
    ed = _editor_with_image(tmp_path, make_scene_image, monkeypatch, cache={"img.jpg": entry})
    ed.quad[0] = [20.0, 20.0]   # actually move a corner
    ed._commit()
    assert ed.cache["img.jpg"]["source"] == "user"


def test_editor_commit_new_image_is_user(tmp_path, make_scene_image, monkeypatch):
    # No prior cache entry -> even an unchanged (auto-detected) quad commits as
    # 'user' because the user is taking ownership of a freshly seen image.
    ed = _editor_with_image(tmp_path, make_scene_image, monkeypatch, cache=None)
    ed._commit()
    assert ed.cache["img.jpg"]["source"] == "user"


def test_editor_to_full_clamps_to_frame(tmp_path, make_scene_image, monkeypatch):
    ed = _editor_with_image(tmp_path, make_scene_image, monkeypatch, cache=None)
    # scale == 1.0 (image smaller than max_dim); out-of-frame input is clamped.
    assert ed._to_full(-5, -5) == [0, 0]
    assert ed._to_full(10_000, 10_000) == [ed.w, ed.h]


def test_editor_save_disk_writes_and_clears_dirty(tmp_path, make_scene_image, monkeypatch):
    ed = _editor_with_image(tmp_path, make_scene_image, monkeypatch, cache=None)
    ed._save_disk()
    assert ed.dirty_disk is False
    assert os.path.exists(ed.cache_path)
    assert "img.jpg" in augment.load_corners_cache(str(tmp_path))


def test_editor_goto_wraps_and_commits(tmp_path, make_scene_image, monkeypatch):
    make_scene_image(str(tmp_path), "a.jpg", w=200, h=260)
    make_scene_image(str(tmp_path), "b.jpg", w=200, h=260)
    monkeypatch.setattr(corners, "detect_paper_quad",
                        lambda im: [[10, 10], [190, 10], [190, 250], [10, 250]])
    ed = corners._CornerEditor(str(tmp_path), max_dim=1100)
    assert ed.idx == 0
    ed._goto(1)
    assert ed.idx == 1
    ed._goto(1)             # wraps back to 0
    assert ed.idx == 0
    assert len(ed.cache) >= 1   # visited images were committed
