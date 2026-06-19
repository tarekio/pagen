"""augment_gen.py — Post-process a detect_gen.py doctr dataset to near-real-life images.

For each entry in labels.json, one of three paths is chosen by probability weights:
  clean  — pass through untouched (model keeps seeing pristine samples)
  scan   — multiply-blend text onto a scan texture (ruled lines / grid / speckle)
  pics   — perspective-warp text onto a blank-page photo, transform polygons identically

Polygon ground truth is preserved exactly: scan path leaves polygons unchanged; pics
path applies cv2.perspectiveTransform with the same homography used to warp the image.

Usage:
  python augment_gen.py -d output/train [--out output/train_aug] [--seed N]
         [--clean-prob 0.1] [--scan-prob 0.45] [--pics-prob 0.45]
         [--pics-dir images/images_pics] [--scan-dir images/images_scan]
         [--workers N]
"""

import argparse
import hashlib
import io
import json
import multiprocessing
import os
import random
import re
import shutil

import cv2
import numpy as np
from PIL import Image

PICS_CORNERS_FILE = "paper_corners.json"

# ---------------------------------------------------------------------------
# Paper-quad detection helpers (Path B bootstrap)
# ---------------------------------------------------------------------------

def _order_quad(pts):
    """Return 4 points in (TL, TR, BR, BL) order."""
    pts = sorted(pts, key=lambda p: p[1])          # sort by y
    top = sorted(pts[:2], key=lambda p: p[0])       # top two, left→right
    bot = sorted(pts[2:], key=lambda p: p[0])       # bottom two, left→right
    return [top[0], top[1], bot[1], bot[0]]


def _best_contour(contours, frame_area):
    """Return the largest contour whose area is 10–97% of the frame, or None."""
    best, best_area = None, 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 0.10 * frame_area <= area <= 0.97 * frame_area and area > best_area:
            best, best_area = cnt, area
    return best


def _quad_from_contour(cnt):
    """Try approxPolyDP → exactly 4 pts; fall back to minAreaRect (always 4 pts)."""
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
    if len(approx) == 4 and cv2.isContourConvex(approx):
        return [p[0].tolist() for p in approx]
    # minAreaRect always gives a 4-point box — robust to shadows inside paper
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    return [p.tolist() for p in box]


def _is_full_frame(quad, w, h, tol=0.98):
    """Return True if the quad bounding box spans ≥tol of both image dimensions.
    Used to reject near-full-image rects produced when a contour covers the whole frame."""
    xs = [p[0] for p in quad]
    ys = [p[1] for p in quad]
    return (max(xs) - min(xs)) >= tol * w and (max(ys) - min(ys)) >= tol * h


def detect_paper_quad(img_bgr):
    """Auto-detect the paper rectangle in a photo.

    Three-pass strategy:
      1. Otsu threshold → largest valid contour → approxPolyDP (4 pts) or minAreaRect
      2. Canny edges → largest valid contour → same quad extraction
      3. Brightness mask (top-N% bright pixels) → same quad extraction
    Returns [[x,y]*4] in TL,TR,BR,BL order, or None if all passes fail.
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
        if _is_full_frame(quad, w, h):
            return None   # reject near-full-image rects; try next pass
        return quad

    # Pass 1: Otsu threshold
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    result = _try_contours(thresh)
    if result:
        return result

    # Pass 2: Canny with low thresholds (general edges)
    edges = cv2.Canny(blur, 30, 100)
    edges = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=2)
    result = _try_contours(edges)
    if result:
        return result

    # Pass 3: border flood-fill — use Canny edges as barriers, flood background
    # from the image border, then invert to get the paper interior.  Robust against
    # carpet texture at the border merging with the paper edge after plain contour finding.
    edges_hard = cv2.Canny(blur, 80, 200)
    edges_hard = cv2.dilate(edges_hard, np.ones((5, 5), np.uint8), iterations=2)
    edges_hard = cv2.morphologyEx(edges_hard, cv2.MORPH_CLOSE, np.ones((40, 40), np.uint8))
    # Barriers = 0, passable = 255; flood from a 1-px border seed
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

    # Pass 4: brightness mask — paper is almost always brighter than its surroundings
    bright_thresh = int(np.percentile(blur, 40))   # top-60% brightest pixels
    _, bright = cv2.threshold(blur, bright_thresh, 255, cv2.THRESH_BINARY)
    result = _try_contours(bright)
    return result   # None if all passes failed


def ensure_corners_cache(pics_dir):
    """Load corners cache, detect missing entries, write back. Returns dict."""
    cache_path = os.path.join(pics_dir, PICS_CORNERS_FILE)
    cache = {}
    if os.path.exists(cache_path):
        with open(cache_path, "r") as f:
            cache = json.load(f)

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
    images = sorted(
        f for f in os.listdir(pics_dir)
        if os.path.splitext(f)[1].lower() in exts
    )

    changed = False
    for fname in images:
        if fname in cache:
            continue
        fpath = os.path.join(pics_dir, fname)
        img = cv2.imread(fpath)
        if img is None:
            print(f"  [warn] cannot read {fpath}, skipping corner detection")
            continue
        quad = detect_paper_quad(img)
        ih, iw = img.shape[:2]
        if quad is None:
            quad = [[0, 0], [iw, 0], [iw, ih], [0, ih]]
            print(f"  [warn] could not detect paper in {fname}, using full-image rect. "
                  f"Edit {cache_path} to fix.")
        else:
            # Round to int and clip to image bounds so H is well-conditioned
            quad = [[max(0, min(iw, int(round(x)))), max(0, min(ih, int(round(y))))]
                    for x, y in quad]
            print(f"  [info] detected paper in {fname}: {quad}")
        cache[fname] = quad
        changed = True

    if changed:
        with open(cache_path, "w") as f:
            json.dump(cache, f, indent=2)
        print(f"Corner cache written: {cache_path}")

    return cache


# ---------------------------------------------------------------------------
# Path A — scan texture composite
# ---------------------------------------------------------------------------

def _apply_scan(page_png_path, scan_imgs):
    """Multiply-blend the page onto a random scan texture. Returns (rgb ndarray, None)
    meaning polygons are unchanged."""
    page = cv2.imread(page_png_path)           # BGR, white bg
    if page is None:
        return None, None
    H, W = page.shape[:2]

    scan_path = random.choice(scan_imgs)
    scan = cv2.imread(scan_path)
    if scan is None:
        return page, None

    scan_resized = cv2.resize(scan, (W, H), interpolation=cv2.INTER_LINEAR)

    # Reduce scan contrast/brightness so texture noise doesn't bleed into text:
    # remap pixel values from [0,255] → [200,255] (subtle paper grain only)
    scan_f = scan_resized.astype(np.float32)
    scan_f = scan_f * (100.0 / 255.0) + 155.0
    scan_f = np.clip(scan_f, 0, 255)

    # Convert page to grayscale mask (0=ink, 255=paper)
    gray = cv2.cvtColor(page, cv2.COLOR_BGR2GRAY)
    # Multiply: texture * (gray/255) — white areas show texture, ink darkens it
    mask = gray.astype(np.float32) / 255.0
    mask3 = np.stack([mask, mask, mask], axis=2)
    out = (scan_f * mask3).astype(np.uint8)
    return out, None   # None → caller keeps existing polygons


# ---------------------------------------------------------------------------
# Path B — photo paper perspective warp
# ---------------------------------------------------------------------------

def _long_short(a, b):
    """Return (longer, shorter) of two values."""
    return (a, b) if a >= b else (b, a)


def _apply_pics(page_png_path, polygons, pic_imgs, corners_cache):
    """Perspective-warp the page onto a blank-paper photo.

    Returns (rgb ndarray, new_polygons) where new_polygons are the original
    polygons transformed by the same homography applied to the image.
    """
    page = cv2.imread(page_png_path)
    if page is None:
        return None, None

    pH, pW = page.shape[:2]

    # Pick a random background photo
    pic_fname = random.choice(list(corners_cache.keys()))
    # Find the full path
    pic_dir = None
    for img_path in pic_imgs:
        if os.path.basename(img_path) == pic_fname:
            pic_dir = img_path
            break
    if pic_dir is None:
        return None, None

    photo = cv2.imread(pic_dir)
    if photo is None:
        return None, None
    fH, fW = photo.shape[:2]

    quad = corners_cache[pic_fname]   # [[x,y]*4] TL,TR,BR,BL
    quad_pts = np.array(quad, dtype=np.float32)

    # Measure quad's width and height spans to decide page orientation
    top_w = np.linalg.norm(quad_pts[1] - quad_pts[0])
    bot_w = np.linalg.norm(quad_pts[2] - quad_pts[3])
    left_h = np.linalg.norm(quad_pts[3] - quad_pts[0])
    right_h = np.linalg.norm(quad_pts[2] - quad_pts[1])
    quad_w = (top_w + bot_w) / 2.0
    quad_h = (left_h + right_h) / 2.0

    # Match page long side → quad long side
    page_long, page_short = _long_short(pH, pW)
    quad_long, quad_short = _long_short(quad_h, quad_w)

    if (page_long == pH) == (quad_long == quad_h):
        # Same orientation: page portrait↔quad tall, or page landscape↔quad wide
        src_pts = np.array([[0, 0], [pW, 0], [pW, pH], [0, pH]], dtype=np.float32)
    else:
        # Rotate page 90° to match quad orientation — rotate src corners
        # (0,0)→(pH,0)→(pH,pW)→(0,pW) in the rotated frame
        src_pts = np.array([[0, pW], [0, 0], [pH, 0], [pH, pW]], dtype=np.float32)

    # Homography: page → photo quad
    H_mat = cv2.getPerspectiveTransform(src_pts, quad_pts)

    # Warp page onto photo (full photo size)
    warped_page = cv2.warpPerspective(page, H_mat, (fW, fH))

    # Build quad mask for compositing
    mask = np.zeros((fH, fW), dtype=np.uint8)
    cv2.fillConvexPoly(mask, quad_pts.astype(np.int32), 255)
    mask3 = np.stack([mask, mask, mask], axis=2)

    # Multiply-blend inside the mask so photo texture/shadow shows through
    page_gray = cv2.cvtColor(warped_page, cv2.COLOR_BGR2GRAY)
    alpha = page_gray.astype(np.float32) / 255.0
    alpha3 = np.stack([alpha, alpha, alpha], axis=2)

    out = photo.copy().astype(np.float32)
    out = np.where(mask3 > 0,
                   out * alpha3,
                   out)
    out = np.clip(out, 0, 255).astype(np.uint8)

    # Transform polygons with the exact same H_mat
    if not polygons:
        return out, polygons

    # Stack all polygon points: shape (N*4, 1, 2)
    all_pts = np.array(polygons, dtype=np.float32).reshape(-1, 1, 2)
    transformed = cv2.perspectiveTransform(all_pts, H_mat)   # (N*4, 1, 2)
    transformed = transformed.reshape(-1, 4, 2)

    new_polys = []
    for quad_r in transformed:
        clipped = [
            [int(np.clip(x, 0, fW)), int(np.clip(y, 0, fH))]
            for x, y in quad_r
        ]
        new_polys.append(clipped)

    return out, new_polys


# ---------------------------------------------------------------------------
# Degradation stage (applied to scan and pics paths, not clean)
# ---------------------------------------------------------------------------

def degrade(img_bgr, rng):
    """Apply a random subset of photometric degradations. Does not move polygons."""
    out = img_bgr.copy()

    # Gaussian blur
    if rng.random() < 0.5:
        k = rng.choice([3, 5])
        out = cv2.GaussianBlur(out, (k, k), 0)

    # Motion blur (horizontal or vertical, small kernel)
    if rng.random() < 0.3:
        k = rng.randint(3, 7)
        if rng.random() < 0.5:
            kernel = np.zeros((k, k), np.float32)
            kernel[k // 2, :] = 1.0 / k
        else:
            kernel = np.zeros((k, k), np.float32)
            kernel[:, k // 2] = 1.0 / k
        out = cv2.filter2D(out, -1, kernel)

    # Resolution loss
    if rng.random() < 0.4:
        H, W = out.shape[:2]
        scale = rng.uniform(0.5, 0.85)
        small = cv2.resize(out, (max(1, int(W * scale)), max(1, int(H * scale))),
                           interpolation=cv2.INTER_LINEAR)
        out = cv2.resize(small, (W, H), interpolation=cv2.INTER_LINEAR)

    # Gaussian noise
    if rng.random() < 0.5:
        noise = np.random.normal(0, rng.uniform(3, 12), out.shape).astype(np.float32)
        out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # Brightness / contrast jitter — kept tight to preserve ink/background separation
    if rng.random() < 0.6:
        alpha = rng.uniform(0.9, 1.1)   # contrast
        beta = rng.uniform(-10, 10)     # brightness
        out = np.clip(out.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

    # Slight color cast
    if rng.random() < 0.3:
        for c in range(3):
            shift = rng.uniform(-10, 10)
            out[:, :, c] = np.clip(out[:, :, c].astype(np.float32) + shift, 0, 255)

    # JPEG recompression artifacts (last, so they persist)
    if rng.random() < 0.5:
        q = int(rng.uniform(55, 90))
        _, enc = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, q])
        out = cv2.imdecode(enc, cv2.IMREAD_COLOR)

    return out


# ---------------------------------------------------------------------------
# Per-entry worker
# ---------------------------------------------------------------------------

def _process_entry(args):
    (img_name, entry, src_images_dir, dst_images_dir,
     clean_prob, scan_prob, scan_imgs, corners_cache, pic_imgs, seed_val) = args

    rng = random.Random(seed_val)
    np.random.seed(seed_val % (2**32))

    src_path = os.path.join(src_images_dir, img_name)
    dst_path = os.path.join(dst_images_dir, img_name)

    polygons = entry.get("polygons", [])

    # Pick path
    r = rng.random()
    if r < clean_prob:
        # Clean: just copy
        if src_images_dir != dst_images_dir:
            shutil.copy2(src_path, dst_path)
        new_polygons = polygons
        img = cv2.imread(dst_path)
        if img is None:
            return None
        h, w = img.shape[:2]
        with open(dst_path, "rb") as f:
            png_bytes = f.read()
        new_hash = hashlib.sha256(png_bytes).hexdigest()
        new_entry = dict(entry)
        new_entry["img_dimensions"] = [w, h]
        new_entry["img_hash"] = new_hash
        new_entry["polygons"] = new_polygons
        return img_name, new_entry

    elif r < clean_prob + scan_prob:
        result_img, _ = _apply_scan(src_path, scan_imgs)
        new_polygons = polygons
    else:
        if not corners_cache or not pic_imgs:
            # Fallback to scan if no pics available
            result_img, _ = _apply_scan(src_path, scan_imgs)
            new_polygons = polygons
        else:
            result_img, new_polygons = _apply_pics(src_path, polygons, pic_imgs, corners_cache)
            if new_polygons is None:
                new_polygons = polygons

    if result_img is None:
        return None

    # Degrade
    result_img = degrade(result_img, rng)

    # Save
    success, enc = cv2.imencode(".png", result_img)
    if not success:
        return None
    png_bytes = enc.tobytes()
    with open(dst_path, "wb") as f:
        f.write(png_bytes)

    h, w = result_img.shape[:2]
    new_entry = dict(entry)
    new_entry["img_dimensions"] = [w, h]
    new_entry["img_hash"] = hashlib.sha256(png_bytes).hexdigest()
    new_entry["polygons"] = new_polygons
    return img_name, new_entry


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Augment a detect_gen.py doctr dataset with realistic noise."
    )
    parser.add_argument("-d", "--dataset", default="output/train",
                        help="Source dataset directory (contains labels.json + images/)")
    parser.add_argument("--out", default=None,
                        help="Output directory (default: augment in place)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clean-prob", type=float, default=0.10)
    parser.add_argument("--scan-prob", type=float, default=0.45)
    parser.add_argument("--pics-prob", type=float, default=0.45)
    parser.add_argument("--pics-dir", default="images/images_pics")
    parser.add_argument("--scan-dir", default="images/images_scan")
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    args = parser.parse_args()

    # Normalise probabilities
    total = args.clean_prob + args.scan_prob + args.pics_prob
    clean_prob = args.clean_prob / total
    scan_prob = args.scan_prob / total

    src_labels = os.path.join(args.dataset, "labels.json")
    src_images = os.path.join(args.dataset, "images")

    if not os.path.exists(src_labels):
        print(f"ERROR: {src_labels} not found.")
        return

    with open(src_labels, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} entries from {src_labels}")

    # Determine destination
    out_dir = args.out if args.out else args.dataset
    dst_images = os.path.join(out_dir, "images")
    os.makedirs(dst_images, exist_ok=True)

    # If out_dir != dataset, copy all images first then augment in dst
    if out_dir != args.dataset:
        print(f"Copying source images to {dst_images} …")
        for fname in os.listdir(src_images):
            s = os.path.join(src_images, fname)
            d = os.path.join(dst_images, fname)
            if not os.path.exists(d):
                shutil.copy2(s, d)

    # Scan textures
    scan_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    scan_imgs = [
        os.path.join(args.scan_dir, f)
        for f in os.listdir(args.scan_dir)
        if os.path.splitext(f)[1].lower() in scan_exts
    ] if os.path.isdir(args.scan_dir) else []
    print(f"Scan textures: {len(scan_imgs)}")

    # Pics — bootstrap corner cache
    corners_cache = {}
    pic_imgs = []
    if os.path.isdir(args.pics_dir):
        print("Bootstrapping paper corner cache …")
        corners_cache = ensure_corners_cache(args.pics_dir)
        pic_imgs = [
            os.path.join(args.pics_dir, f)
            for f in corners_cache
        ]
    print(f"Photo backgrounds: {len(pic_imgs)}")

    # Build tasks
    random.seed(args.seed)
    tasks = []
    for i, (img_name, entry) in enumerate(data.items()):
        entry_seed = (args.seed * 1000003 + i) % (2**31)
        tasks.append((
            img_name, entry, src_images, dst_images,
            clean_prob, scan_prob, scan_imgs, corners_cache, pic_imgs, entry_seed
        ))

    # Process
    n_workers = max(1, min(args.workers, len(tasks)))
    new_data = {}
    done = 0
    total_entries = len(tasks)

    dst_labels = os.path.join(out_dir, "labels.json")
    with open(dst_labels, "w", encoding="utf-8") as out_f:
        out_f.write("{\n")
        first = [True]

        def _emit(key, value):
            sep = "" if first[0] else ",\n"
            out_f.write(sep + json.dumps(key) + ": " + json.dumps(value, ensure_ascii=False))
            first[0] = False

        with multiprocessing.Pool(n_workers) as pool:
            try:
                for result in pool.imap_unordered(_process_entry, tasks):
                    if result is not None:
                        img_name, new_entry = result
                        _emit(img_name, new_entry)
                    done += 1
                    if done % 50 == 0 or done == total_entries:
                        print(f"  {done}/{total_entries} done")
            except KeyboardInterrupt:
                print("\nInterrupted — terminating workers …")
                pool.terminate()
                pool.join()
            finally:
                out_f.write("\n}\n")

    print(f"Done. labels.json written to {dst_labels}")


if __name__ == "__main__":
    main()
