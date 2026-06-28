"""Unit tests for pagen.augment image ops (fast: tiny synthetic ndarrays).

Script-agnostic — these operate on rendered-page pixels, not on text.
"""

import os
import random

import numpy as np

from pagen import augment
from pagen.augment import AugmentContext


# ---------------------------------------------------------------------------
# degrade
# ---------------------------------------------------------------------------

def test_degrade_preserves_shape_and_dtype(page_bgr):
    for seed in range(5):
        out = augment.degrade(page_bgr, random.Random(seed))
        assert out.shape == page_bgr.shape
        assert out.dtype == np.uint8


def test_degrade_caps_keep_thin_strokes_legible(monkeypatch):
    """The blur/downsample/jpeg caps that keep small Arabic text readable.

    A per-pixel darkness metric is too noisy (the worst washout only shows once
    the pics-path warp pre-shrinks the page), so we pin the parameters directly:
    spy on every cv2 op degrade() performs across many seeds and assert it never
    reaches into the obliterating range.  Old params (Gaussian k=5, downsample
    to 0.5x, JPEG q55) violate every one of these bounds.
    """
    import cv2

    ksizes, shrink_ratios, jpeg_qs = [], [], []

    real_gauss, real_resize, real_imencode = cv2.GaussianBlur, cv2.resize, cv2.imencode

    def spy_gauss(src, ksize, sigmaX, *a, **k):
        ksizes.append(ksize[0])
        return real_gauss(src, ksize, sigmaX, *a, **k)

    def spy_resize(src, dsize, *a, **k):
        if dsize[0] < src.shape[1]:           # the shrink half of a downsample
            shrink_ratios.append(dsize[0] / src.shape[1])
        return real_resize(src, dsize, *a, **k)

    def spy_imencode(ext, img, params=None):
        if params and cv2.IMWRITE_JPEG_QUALITY in params:
            jpeg_qs.append(params[params.index(cv2.IMWRITE_JPEG_QUALITY) + 1])
        return real_imencode(ext, img, params)

    monkeypatch.setattr(augment.cv2, "GaussianBlur", spy_gauss)
    monkeypatch.setattr(augment.cv2, "resize", spy_resize)
    monkeypatch.setattr(augment.cv2, "imencode", spy_imencode)

    img = np.full((200, 280, 3), 255, dtype=np.uint8)
    for seed in range(300):
        augment.degrade(img, random.Random(seed))

    assert ksizes and shrink_ratios and jpeg_qs   # all three branches exercised
    assert max(ksizes) <= 3                        # no k=5 Gaussian
    assert min(shrink_ratios) >= 0.7 - 1e-6        # downsample floored at 0.7x
    assert min(jpeg_qs) >= 65                       # JPEG quality floored at 65


# ---------------------------------------------------------------------------
# augment_page path selection
# ---------------------------------------------------------------------------

def test_augment_page_clean_path_returns_unchanged(page_bgr):
    ctx = AugmentContext(clean_prob=1.0, scan_prob=0.0)
    polys = [[[0, 0], [1, 0], [1, 1], [0, 1]]]
    out, new_polys = augment.augment_page(page_bgr, polys, ctx, random.Random(0))
    assert out is page_bgr           # clean path is a passthrough
    assert new_polys == polys


def test_augment_page_scan_path(page_bgr, make_scene_image, tmp_path):
    scan = make_scene_image(str(tmp_path), "texture.jpg")
    ctx = AugmentContext(scan_imgs=[scan], clean_prob=0.0, scan_prob=1.0)
    polys = [[[0, 0], [1, 0], [1, 1], [0, 1]]]
    out, new_polys = augment.augment_page(page_bgr, polys, ctx, random.Random(0))
    assert out.shape == page_bgr.shape
    assert new_polys == polys        # scan path doesn't move polygons


def test_augment_page_scan_path_without_textures_still_degrades(page_bgr):
    ctx = AugmentContext(scan_imgs=[], clean_prob=0.0, scan_prob=1.0)
    out, new_polys = augment.augment_page(page_bgr, [], ctx, random.Random(0))
    assert out.shape == page_bgr.shape


def test_augment_page_pics_path_transforms_polygons(page_bgr, make_scene_image, tmp_path):
    path = make_scene_image(str(tmp_path), "bg.jpg", w=240, h=300)
    fname = os.path.basename(path)
    quad = [[20, 20], [220, 20], [220, 280], [20, 280]]
    ctx = AugmentContext(
        pic_imgs=[path],
        corners_cache={fname: augment._make_entry(quad, "auto")},
        clean_prob=0.0,
        scan_prob=0.0,   # force pics path
    )
    polys = [[[5, 5], [15, 5], [15, 15], [5, 15]]]
    out, new_polys = augment.augment_page(page_bgr, polys, ctx, random.Random(0))
    assert isinstance(out, np.ndarray)
    assert len(new_polys) == len(polys)
    assert all(len(p) == 4 for p in new_polys)


def test_augment_page_pics_path_empty_polygons(page_bgr, make_scene_image, tmp_path):
    path = make_scene_image(str(tmp_path), "bg2.jpg", w=240, h=300)
    fname = os.path.basename(path)
    quad = [[20, 20], [220, 20], [220, 280], [20, 280]]
    ctx = AugmentContext(
        pic_imgs=[path],
        corners_cache={fname: augment._make_entry(quad, "auto")},
        clean_prob=0.0,
        scan_prob=0.0,
    )
    out, new_polys = augment.augment_page(page_bgr, [], ctx, random.Random(1))
    assert isinstance(out, np.ndarray)
    assert new_polys == []


def test_apply_pics_keeps_portrait_page_upright_on_landscape_paper(tmp_path):
    """Regression: a portrait page warped onto a landscape paper quad must stay
    upright, never turned 90 degrees.

    A thin centered vertical ink stripe is flip- and jitter-invariant but
    rotation-sensitive: upright it stays taller than wide; the old
    orientation-matching branch rotated it to span the paper's long (horizontal)
    axis, making it wider than tall.
    """
    import cv2

    # Portrait page (H > W) with a thin centered vertical black stripe.
    pH, pW = 300, 200
    page = np.full((pH, pW, 3), 255, dtype=np.uint8)
    page[:, pW // 2 - 8: pW // 2 + 8] = 0

    # Plain photo carrying a clearly landscape paper quad (width >> height).
    photo = np.full((400, 700, 3), 255, dtype=np.uint8)
    pic_path = str(tmp_path / "paper.png")
    cv2.imwrite(pic_path, photo)
    quad = [[60, 120], [640, 120], [640, 300], [60, 300]]   # w=580 > h=180
    cache = {"paper.png": augment._make_entry(quad, "auto")}

    out, _ = augment._apply_pics(page, [], [pic_path], cache, random.Random(0))

    gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
    ys, xs = np.where(gray < 128)
    assert xs.size > 0, "page ink should be composited onto the photo"
    ink_w = xs.max() - xs.min()
    ink_h = ys.max() - ys.min()
    assert ink_h > ink_w, "vertical stripe became horizontal -> page was rotated"


def test_augment_page_pics_path_fallback_to_degrade(page_bgr):
    # No corners/pics and no scan textures -> just degrade, polygons unchanged.
    ctx = AugmentContext(clean_prob=0.0, scan_prob=0.0)
    polys = [[[0, 0], [1, 0], [1, 1], [0, 1]]]
    out, new_polys = augment.augment_page(page_bgr, polys, ctx, random.Random(0))
    assert out.shape == page_bgr.shape
    assert new_polys == polys
