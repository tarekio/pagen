"""Augmentation pipeline: scan texture, photo perspective warp, degradation.

``augment_page()`` is the fused entry point called per page in the pipeline.
It accepts a decoded BGR ndarray (the rendered page) and returns
(augmented_bgr, new_polygons).

The paper-corner detection and cache helpers are also exposed here so that
``corners.py`` can import them directly.

PICS_CORNERS_FILE, detect_paper_quad, _order_quad kept as public names.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

PICS_CORNERS_FILE = "paper_corners.json"


# ---------------------------------------------------------------------------
# Quad geometry helpers
# ---------------------------------------------------------------------------

def _order_quad(pts):
    """Return 4 points in (TL, TR, BR, BL) order."""
    pts = sorted(pts, key=lambda p: p[1])
    top = sorted(pts[:2], key=lambda p: p[0])
    bot = sorted(pts[2:], key=lambda p: p[0])
    return [top[0], top[1], bot[1], bot[0]]


def _best_contour(contours, frame_area):
    best, best_area = None, 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 0.10 * frame_area <= area <= 0.97 * frame_area and area > best_area:
            best, best_area = cnt, area
    return best


def _quad_from_contour(cnt):
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
    if len(approx) == 4 and cv2.isContourConvex(approx):
        return [p[0].tolist() for p in approx]
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    return [p.tolist() for p in box]


def _is_full_frame(quad, w, h, tol=0.98):
    xs = [p[0] for p in quad]
    ys = [p[1] for p in quad]
    return (max(xs) - min(xs)) >= tol * w and (max(ys) - min(ys)) >= tol * h


# ---------------------------------------------------------------------------
# Paper quad auto-detection
# ---------------------------------------------------------------------------

def detect_paper_quad(img_bgr):
    """Auto-detect the paper rectangle in a photo.

    Three-pass strategy: Otsu threshold, Canny edges, border flood-fill,
    brightness mask.  Returns [[x,y]*4] in TL,TR,BR,BL order, or None.
    """
    h, w = img_bgr.shape[:2]
    frame_area = h * w
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    close_k = np.ones((25, 25), np.uint8)

    def _try_contours(binary):
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_k)
        cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnt = _best_contour(cnts, frame_area)
        if cnt is None:
            return None
        quad = _order_quad(_quad_from_contour(cnt))
        return None if _is_full_frame(quad, w, h) else quad

    # Pass 1: Otsu
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    result = _try_contours(thresh)
    if result:
        return result

    # Pass 2: Canny
    edges = cv2.Canny(blur, 30, 100)
    edges = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=2)
    result = _try_contours(edges)
    if result:
        return result

    # Pass 3: border flood-fill
    edges_hard = cv2.Canny(blur, 80, 200)
    edges_hard = cv2.dilate(edges_hard, np.ones((5, 5), np.uint8), iterations=2)
    edges_hard = cv2.morphologyEx(edges_hard, cv2.MORPH_CLOSE, np.ones((40, 40), np.uint8))
    barriers = cv2.bitwise_not(edges_hard)
    padded = cv2.copyMakeBorder(barriers, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=255)
    flood_mask = np.zeros((h + 4, w + 4), np.uint8)
    cv2.floodFill(padded, flood_mask, (0, 0), 0)
    background = (padded[1:h+1, 1:w+1] == 0).astype(np.uint8) * 255
    paper_interior = cv2.bitwise_not(background)
    paper_interior = cv2.morphologyEx(paper_interior, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    cnts, _ = cv2.findContours(paper_interior, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = _best_contour(cnts, frame_area)
    if cnt is not None:
        quad = _order_quad(_quad_from_contour(cnt))
        if not _is_full_frame(quad, w, h):
            return quad

    # Pass 4: brightness mask
    bright_thresh = int(np.percentile(blur, 40))
    _, bright = cv2.threshold(blur, bright_thresh, 255, cv2.THRESH_BINARY)
    return _try_contours(bright)


# ---------------------------------------------------------------------------
# Corner cache
#
# Each entry is {"quad": [[x,y]*4], "source": "auto"|"user"}.  "user" entries
# come from the interactive editor and are NEVER overwritten by auto-detection.
# The legacy bare-list format ({fname: [[x,y]*4]}) is still read; such entries
# are treated as "auto" provenance.
# ---------------------------------------------------------------------------

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


def quad_of(entry):
    """Return the [[x,y]*4] quad from a cache entry (new dict or legacy list)."""
    return entry["quad"] if isinstance(entry, dict) else entry


def source_of(entry):
    """Return provenance ('auto'/'user') of a cache entry; legacy lists are 'auto'."""
    return entry.get("source", "auto") if isinstance(entry, dict) else "auto"


def _make_entry(quad, source):
    return {"quad": quad, "source": source}


def load_corners_cache(pics_dir: str) -> dict:
    """Load the cache, normalizing every entry to {'quad', 'source'} form."""
    cache_path = os.path.join(pics_dir, PICS_CORNERS_FILE)
    cache = {}
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            raw = json.load(f)
        for fname, value in raw.items():
            cache[fname] = _make_entry(quad_of(value), source_of(value))
    return cache


def write_corners_cache(pics_dir: str, cache: dict) -> None:
    """Persist the normalized cache to paper_corners.json."""
    cache_path = os.path.join(pics_dir, PICS_CORNERS_FILE)
    with open(cache_path, "w") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)


def ensure_corners_cache(pics_dir: str) -> dict:
    """Reconcile the corner cache with the images on disk.

    Reads both the cache and the image list.  Images already in the cache are
    left exactly as-is — existing entries (and especially ``source: "user"``
    ones) are never re-detected or overwritten.  Only genuinely new images are
    auto-detected, appended with ``source: "auto"``, saved, and reported.  When
    nothing is new the file is not touched at all.

    Returns the full normalized cache {fname: {'quad', 'source'}}.
    """
    cache = load_corners_cache(pics_dir)
    images = sorted(f for f in os.listdir(pics_dir) if os.path.splitext(f)[1].lower() in IMG_EXTS)

    new_images = [f for f in images if f not in cache]
    detected = []
    for fname in new_images:
        img = cv2.imread(os.path.join(pics_dir, fname))
        if img is None:
            print(f"  [warn] cannot read {fname}, skipping")
            continue
        ih, iw = img.shape[:2]
        quad = detect_paper_quad(img)
        if quad is None:
            quad = [[0, 0], [iw, 0], [iw, ih], [0, ih]]
            print(f"  [warn] could not detect paper in {fname}, using full-image rect")
        else:
            quad = [[max(0, min(iw, int(round(x)))), max(0, min(ih, int(round(y))))]
                    for x, y in quad]
        cache[fname] = _make_entry(quad, "auto")
        detected.append(fname)

    if detected:
        write_corners_cache(pics_dir, cache)
        print(f"  [info] auto-detected and saved corners for {len(detected)} new image(s) "
              f"(existing entries untouched):")
        for fname in detected:
            print(f"      + {fname}")

    return cache


# ---------------------------------------------------------------------------
# AugmentContext — precomputed resources passed to workers
# ---------------------------------------------------------------------------

@dataclass
class AugmentContext:
    scan_imgs: list[str] = field(default_factory=list)
    corners_cache: dict = field(default_factory=dict)
    pic_imgs: list[str] = field(default_factory=list)
    clean_prob: float = 0.10
    scan_prob: float = 0.45


# ---------------------------------------------------------------------------
# Path A — scan texture composite
# ---------------------------------------------------------------------------

def _apply_scan(page_bgr: np.ndarray, scan_imgs: list[str], rng: random.Random) -> np.ndarray:
    """Multiply-blend the page onto a random scan texture."""
    H, W = page_bgr.shape[:2]

    scan_path = rng.choice(scan_imgs)
    scan = cv2.imread(scan_path)
    if scan is None:
        return page_bgr

    if rng.random() < 0.5:
        scan = cv2.flip(scan, 1)
    if rng.random() < 0.3:
        scan = cv2.flip(scan, 0)

    sH, sW = scan.shape[:2]
    def _ri(dim):
        return int(rng.uniform(0.02, 0.08) * dim)
    src = np.array([
        [_ri(sW),        _ri(sH)],
        [sW - _ri(sW),   _ri(sH)],
        [sW - _ri(sW),   sH - _ri(sH)],
        [_ri(sW),        sH - _ri(sH)],
    ], dtype=np.float32)
    dst = np.array([[0, 0], [sW, 0], [sW, sH], [0, sH]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    scan = cv2.warpPerspective(scan, M, (sW, sH))

    scan_resized = cv2.resize(scan, (W, H), interpolation=cv2.INTER_LINEAR)
    scan_f = scan_resized.astype(np.float32) * (100.0 / 255.0) + 155.0
    scan_f = np.clip(scan_f, 0, 255)

    gray = cv2.cvtColor(page_bgr, cv2.COLOR_BGR2GRAY)
    mask = gray.astype(np.float32) / 255.0
    mask3 = np.stack([mask, mask, mask], axis=2)
    return (scan_f * mask3).astype(np.uint8)


# ---------------------------------------------------------------------------
# Path B — photo paper perspective warp
# ---------------------------------------------------------------------------

def _apply_pics(
    page_bgr: np.ndarray,
    polygons: list,
    pic_imgs: list[str],
    corners_cache: dict,
    rng: random.Random,
) -> tuple[np.ndarray, list]:
    """Perspective-warp the page onto a blank-paper photo.

    Returns (augmented_bgr, new_polygons).
    """
    pH, pW = page_bgr.shape[:2]

    pic_fname = rng.choice(list(corners_cache.keys()))
    pic_dir = next((p for p in pic_imgs if os.path.basename(p) == pic_fname), None)
    if pic_dir is None:
        return page_bgr, polygons

    photo = cv2.imread(pic_dir)
    if photo is None:
        return page_bgr, polygons
    fH, fW = photo.shape[:2]

    quad_raw = np.array(quad_of(corners_cache[pic_fname]), dtype=np.float32)
    xs, ys = quad_raw[:, 0], quad_raw[:, 1]
    pad_x = (xs.max() - xs.min()) * rng.uniform(0.10, 0.15)
    pad_y = (ys.max() - ys.min()) * rng.uniform(0.10, 0.15)
    cx0 = max(0, int(xs.min() - pad_x))
    cy0 = max(0, int(ys.min() - pad_y))
    cx1 = min(fW, int(xs.max() + pad_x))
    cy1 = min(fH, int(ys.max() + pad_y))
    photo = photo[cy0:cy1, cx0:cx1]
    fH, fW = photo.shape[:2]

    flip_h = rng.random() < 0.5
    flip_v = rng.random() < 0.3
    if flip_h:
        photo = cv2.flip(photo, 1)
    if flip_v:
        photo = cv2.flip(photo, 0)

    def _rand_inset(dim):
        return int(rng.uniform(0.02, 0.08) * dim)

    bg_src = np.array([
        [_rand_inset(fW),        _rand_inset(fH)],
        [fW - _rand_inset(fW),   _rand_inset(fH)],
        [fW - _rand_inset(fW),   fH - _rand_inset(fH)],
        [_rand_inset(fW),        fH - _rand_inset(fH)],
    ], dtype=np.float32)
    bg_dst = np.array([[0, 0], [fW, 0], [fW, fH], [0, fH]], dtype=np.float32)
    H_bg = cv2.getPerspectiveTransform(bg_src, bg_dst)
    photo = cv2.warpPerspective(photo, H_bg, (fW, fH))

    quad_pts = quad_raw - np.array([cx0, cy0], dtype=np.float32)
    if flip_h:
        quad_pts[:, 0] = fW - quad_pts[:, 0]
    if flip_v:
        quad_pts[:, 1] = fH - quad_pts[:, 1]
    quad_pts = cv2.perspectiveTransform(quad_pts.reshape(-1, 1, 2), H_bg).reshape(-1, 2)
    quad_pts = np.array(_order_quad(quad_pts.tolist()), dtype=np.float32)

    centroid = quad_pts.mean(axis=0)
    for i in range(4):
        t = rng.uniform(0.0, 0.05)
        quad_pts[i] = quad_pts[i] + t * (centroid - quad_pts[i])

    # Map the page onto the paper quad upright (page TL,TR,BR,BL -> quad
    # TL,TR,BR,BL).  We never rotate the page 90 degrees to match the paper's
    # orientation: rotated text is wrong for OCR training even when it avoids
    # stretching, so a portrait page on a landscape paper is allowed to distort
    # instead of being turned sideways.
    src_pts = np.array([[0, 0], [pW, 0], [pW, pH], [0, pH]], dtype=np.float32)
    H_mat = cv2.getPerspectiveTransform(src_pts, quad_pts)
    warped_page = cv2.warpPerspective(page_bgr, H_mat, (fW, fH), borderValue=(255, 255, 255))

    mask = np.zeros((fH, fW), dtype=np.uint8)
    cv2.fillConvexPoly(mask, quad_pts.astype(np.int32), 255)
    mask3 = np.stack([mask, mask, mask], axis=2)

    page_gray = cv2.cvtColor(warped_page, cv2.COLOR_BGR2GRAY)
    alpha = page_gray.astype(np.float32) / 255.0
    alpha3 = np.stack([alpha, alpha, alpha], axis=2)

    out = photo.copy().astype(np.float32)
    out = np.where(mask3 > 0, out * alpha3, out)
    out = np.clip(out, 0, 255).astype(np.uint8)

    if not polygons:
        return out, polygons

    all_pts = np.array(polygons, dtype=np.float32).reshape(-1, 1, 2)
    transformed = cv2.perspectiveTransform(all_pts, H_mat).reshape(-1, 4, 2)
    new_polys = [
        [[int(np.clip(x, 0, fW)), int(np.clip(y, 0, fH))] for x, y in quad_r]
        for quad_r in transformed
    ]
    return out, new_polys


# ---------------------------------------------------------------------------
# Degradation
# ---------------------------------------------------------------------------

def degrade(img_bgr: np.ndarray, rng: random.Random) -> np.ndarray:
    """Apply a random subset of photometric degradations."""
    out = img_bgr.copy()

    if rng.random() < 0.5:
        # k=5 obliterates thin (~1-2px) Arabic strokes on small text, washing
        # black ink to mid-gray; k=3 keeps it legible while still softening.
        out = cv2.GaussianBlur(out, (3, 3), 0)

    if rng.random() < 0.3:
        k = rng.randint(3, 7)
        if rng.random() < 0.5:
            kernel = np.zeros((k, k), np.float32)
            kernel[k // 2, :] = 1.0 / k
        else:
            kernel = np.zeros((k, k), np.float32)
            kernel[:, k // 2] = 1.0 / k
        out = cv2.filter2D(out, -1, kernel)

    if rng.random() < 0.4:
        H, W = out.shape[:2]
        # Floor at 0.7: 0.5x downsampling (often stacked on the pics-path warp,
        # which already shrinks the page) destroys thin strokes and is the main
        # cause of unreadable washed-out text.
        scale = rng.uniform(0.7, 0.9)
        small = cv2.resize(out, (max(1, int(W * scale)), max(1, int(H * scale))), interpolation=cv2.INTER_LINEAR)
        out = cv2.resize(small, (W, H), interpolation=cv2.INTER_LINEAR)

    if rng.random() < 0.5:
        noise = np.random.normal(0, rng.uniform(3, 12), out.shape).astype(np.float32)
        out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    if rng.random() < 0.6:
        alpha = rng.uniform(0.9, 1.1)
        beta = rng.uniform(-10, 10)
        out = np.clip(out.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

    if rng.random() < 0.3:
        for c in range(3):
            shift = rng.uniform(-10, 10)
            out[:, :, c] = np.clip(out[:, :, c].astype(np.float32) + shift, 0, 255)

    if rng.random() < 0.5:
        q = int(rng.uniform(65, 92))
        _, enc = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, q])
        out = cv2.imdecode(enc, cv2.IMREAD_COLOR)

    return out


# ---------------------------------------------------------------------------
# Fused entry point
# ---------------------------------------------------------------------------

def augment_page(
    page_bgr: np.ndarray,
    polygons: list,
    ctx: AugmentContext,
    rng: random.Random,
) -> tuple[np.ndarray, list]:
    """Choose a path (clean/scan/pics), augment, degrade, return (img, polygons).

    Clean path: returns page unchanged (no degrade).
    Scan/pics paths: augment then degrade.
    """
    r = rng.random()

    if r < ctx.clean_prob:
        return page_bgr, polygons

    if r < ctx.clean_prob + ctx.scan_prob:
        if ctx.scan_imgs:
            result = _apply_scan(page_bgr, ctx.scan_imgs, rng)
        else:
            result = page_bgr
        return degrade(result, rng), polygons

    # pics path
    if ctx.corners_cache and ctx.pic_imgs:
        result, new_polys = _apply_pics(page_bgr, polygons, ctx.pic_imgs, ctx.corners_cache, rng)
        return degrade(result, rng), new_polys
    elif ctx.scan_imgs:
        result = _apply_scan(page_bgr, ctx.scan_imgs, rng)
        return degrade(result, rng), polygons
    else:
        return degrade(page_bgr, rng), polygons
