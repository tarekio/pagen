"""Unit tests for pagen.augment corner-cache + geometry (script-agnostic).

Image ops live in test_augment_image.py; here we cover the JSON cache
reconciliation rules, which are central to the corners workflow.
"""

import json

from pagen import augment


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def test_order_quad_returns_tl_tr_br_bl():
    # A known rectangle, points given out of order.
    pts = [[10, 10], [0, 0], [0, 10], [10, 0]]
    assert augment._order_quad(pts) == [[0, 0], [10, 0], [10, 10], [0, 10]]


def test_is_full_frame_true_and_false():
    full = [[0, 0], [100, 0], [100, 100], [0, 100]]
    inset = [[20, 20], [80, 20], [80, 80], [20, 80]]
    assert augment._is_full_frame(full, 100, 100)
    assert not augment._is_full_frame(inset, 100, 100)


# ---------------------------------------------------------------------------
# Cache entry accessors (new dict + legacy list)
# ---------------------------------------------------------------------------

def test_quad_of_and_source_of_dict():
    entry = {"quad": [[0, 0]], "source": "user"}
    assert augment.quad_of(entry) == [[0, 0]]
    assert augment.source_of(entry) == "user"


def test_quad_of_and_source_of_legacy_list():
    legacy = [[0, 0], [1, 0], [1, 1], [0, 1]]
    assert augment.quad_of(legacy) == legacy
    assert augment.source_of(legacy) == "auto"


# ---------------------------------------------------------------------------
# load / write round trip + normalization
# ---------------------------------------------------------------------------

def test_load_corners_cache_missing_file(tmp_path):
    assert augment.load_corners_cache(str(tmp_path)) == {}


def test_load_corners_cache_normalizes_legacy(tmp_path):
    raw = {"a.jpg": [[0, 0], [1, 0], [1, 1], [0, 1]]}
    (tmp_path / augment.PICS_CORNERS_FILE).write_text(json.dumps(raw))
    cache = augment.load_corners_cache(str(tmp_path))
    assert cache["a.jpg"] == {"quad": raw["a.jpg"], "source": "auto"}


def test_write_then_load_round_trip(tmp_path):
    cache = {"a.jpg": augment._make_entry([[0, 0], [1, 0], [1, 1], [0, 1]], "user")}
    augment.write_corners_cache(str(tmp_path), cache)
    assert augment.load_corners_cache(str(tmp_path)) == cache


# ---------------------------------------------------------------------------
# ensure_corners_cache reconciliation
# ---------------------------------------------------------------------------

def test_ensure_appends_only_new_images(tmp_path, make_scene_image, monkeypatch):
    make_scene_image(str(tmp_path), "new.jpg")
    monkeypatch.setattr(augment, "detect_paper_quad",
                        lambda img: [[1, 1], [9, 1], [9, 9], [1, 9]])
    cache = augment.ensure_corners_cache(str(tmp_path))
    assert cache["new.jpg"]["source"] == "auto"
    assert cache["new.jpg"]["quad"] == [[1, 1], [9, 1], [9, 9], [1, 9]]


def test_ensure_never_overwrites_existing_user_entry(tmp_path, make_scene_image, monkeypatch):
    make_scene_image(str(tmp_path), "img.jpg")
    user_entry = augment._make_entry([[2, 2], [3, 2], [3, 3], [2, 3]], "user")
    augment.write_corners_cache(str(tmp_path), {"img.jpg": user_entry})
    # detection would return something different — must be ignored for existing.
    monkeypatch.setattr(augment, "detect_paper_quad", lambda img: [[0, 0], [9, 0], [9, 9], [0, 9]])
    cache = augment.ensure_corners_cache(str(tmp_path))
    assert cache["img.jpg"] == user_entry


def test_ensure_does_not_rewrite_when_nothing_new(tmp_path, make_scene_image, monkeypatch):
    make_scene_image(str(tmp_path), "img.jpg")
    augment.write_corners_cache(
        str(tmp_path), {"img.jpg": augment._make_entry([[0, 0], [1, 0], [1, 1], [0, 1]], "auto")}
    )
    calls = []
    monkeypatch.setattr(augment, "write_corners_cache",
                        lambda d, c: calls.append(1))
    augment.ensure_corners_cache(str(tmp_path))
    assert calls == []   # steady state leaves the file untouched


def test_ensure_undetectable_falls_back_to_full_rect(tmp_path, make_scene_image, monkeypatch):
    path = make_scene_image(str(tmp_path), "blank.jpg", w=50, h=60)
    monkeypatch.setattr(augment, "detect_paper_quad", lambda img: None)
    cache = augment.ensure_corners_cache(str(tmp_path))
    assert cache["blank.jpg"]["quad"] == [[0, 0], [50, 0], [50, 60], [0, 60]]
