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


def test_augment_page_pics_path_fallback_to_degrade(page_bgr):
    # No corners/pics and no scan textures -> just degrade, polygons unchanged.
    ctx = AugmentContext(clean_prob=0.0, scan_prob=0.0)
    polys = [[[0, 0], [1, 0], [1, 1], [0, 1]]]
    out, new_polys = augment.augment_page(page_bgr, polys, ctx, random.Random(0))
    assert out.shape == page_bgr.shape
    assert new_polys == polys
