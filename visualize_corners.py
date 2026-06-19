"""Visualize paper_corners.json detections overlaid on the source images.

Usage:
  python visualize_corners.py [--pics-dir images/images_pics] [--out corners_vis]
"""

import argparse
import json
import os

import cv2
import numpy as np

PICS_CORNERS_FILE = "paper_corners.json"


def draw_quad(img, quad, color=(0, 255, 0), thickness=3):
    pts = np.array(quad, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(img, [pts], isClosed=True, color=color, thickness=thickness)
    labels = ["TL", "TR", "BR", "BL"]
    for i, (x, y) in enumerate(quad):
        cv2.circle(img, (x, y), 8, (0, 0, 255), -1)
        cv2.putText(img, labels[i], (x + 10, y + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    return img


def main():
    parser = argparse.ArgumentParser(description="Visualize paper corner detections.")
    parser.add_argument("--pics-dir", default="images/images_pics")
    parser.add_argument("--out", default="corners_vis", help="Output directory for overlays")
    args = parser.parse_args()

    cache_path = os.path.join(args.pics_dir, PICS_CORNERS_FILE)
    if not os.path.exists(cache_path):
        print(f"ERROR: {cache_path} not found. Run augment_gen.py first to build the cache.")
        return

    with open(cache_path) as f:
        cache = json.load(f)

    os.makedirs(args.out, exist_ok=True)

    for fname, quad in sorted(cache.items()):
        fpath = os.path.join(args.pics_dir, fname)
        img = cv2.imread(fpath)
        if img is None:
            print(f"  [skip] {fname}: cannot read image")
            continue

        # Scale down for display if large
        h, w = img.shape[:2]
        max_dim = 1200
        scale = min(1.0, max_dim / max(h, w))
        if scale < 1.0:
            disp = cv2.resize(img, (int(w * scale), int(h * scale)))
            quad_scaled = [[int(x * scale), int(y * scale)] for x, y in quad]
        else:
            disp = img.copy()
            quad_scaled = quad

        draw_quad(disp, quad_scaled)

        out_path = os.path.join(args.out, fname.rsplit(".", 1)[0] + ".jpg")
        cv2.imwrite(out_path, disp)
        print(f"  {fname}: {quad} → {out_path}")

    print(f"\nSaved {len(cache)} overlays to {args.out}/")


if __name__ == "__main__":
    main()
