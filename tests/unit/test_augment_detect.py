"""Unit tests for pagen.augment.detect_paper_quad (synthetic scenes).

Script-agnostic CV logic.  We assert structural properties (a 4-point inset
quad on a clear scene, None on a degenerate one) rather than exact corners,
which depend on contour approximation.
"""

import numpy as np

from pagen import augment


def _scene_with_paper(W=400, H=320, inset=40):
    """Dark background with a bright white 'paper' rectangle in the middle."""
    img = np.full((H, W, 3), 30, dtype=np.uint8)
    img[inset:H - inset, inset:W - inset] = 245
    return img


def test_detect_returns_inset_quad_on_clear_scene():
    quad = augment.detect_paper_quad(_scene_with_paper())
    assert quad is not None
    assert len(quad) == 4
    H, W = 320, 400
    for x, y in quad:
        assert 0 <= x <= W and 0 <= y <= H
    # Not the full frame: detect_paper_quad returns None for full-frame quads,
    # so a non-None result is necessarily inset.
    xs = [p[0] for p in quad]
    ys = [p[1] for p in quad]
    assert (max(xs) - min(xs)) < 0.98 * W
    assert (max(ys) - min(ys)) < 0.98 * H


def test_detect_returns_none_on_uniform_image():
    blank = np.zeros((200, 200, 3), dtype=np.uint8)
    assert augment.detect_paper_quad(blank) is None


# ---------------------------------------------------------------------------
# Contour helpers
# ---------------------------------------------------------------------------

def test_best_contour_picks_largest_in_range():
    import cv2
    frame_area = 1000 * 1000
    small = np.array([[[10, 10]], [[20, 10]], [[20, 20]], [[10, 20]]], dtype=np.int32)
    # ~0.25 of frame -> within [0.10, 0.97]
    big = np.array([[[0, 0]], [[500, 0]], [[500, 500]], [[0, 500]]], dtype=np.int32)
    chosen = augment._best_contour([small, big], frame_area)
    assert chosen is big
    assert cv2.contourArea(chosen) == cv2.contourArea(big)


def test_best_contour_none_when_all_out_of_range():
    frame_area = 100 * 100
    tiny = np.array([[[0, 0]], [[2, 0]], [[2, 2]], [[0, 2]]], dtype=np.int32)
    assert augment._best_contour([tiny], frame_area) is None


def test_quad_from_contour_square_returns_four_points():
    cnt = np.array([[[0, 0]], [[100, 0]], [[100, 100]], [[0, 100]]], dtype=np.int32)
    quad = augment._quad_from_contour(cnt)
    assert len(quad) == 4
