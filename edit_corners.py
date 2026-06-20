"""edit_corners.py — Interactive viewer/editor for paper-corner detections.

Generic over any directory of paper photos plus its paper_corners.json cache
(the same cache augment_gen.py's pics path consumes). Shows one image at a time,
overlays the four corner handles, and lets you drag them to fix bad detections.

Usage:
  python edit_corners.py [--pics-dir images/images_pics] [--max-dim 1100]

Controls:
  drag handle   move a corner
  n / →         next image            p / ←   previous image
  r             re-run auto-detection  f      reset to full-image rect
  s             save cache to disk      q/Esc  save and quit

Corners are stored [[x,y]*4] in TL,TR,BR,BL order, keyed by filename. Edits are
kept in memory and committed (re-ordered + clipped) when you switch image, save,
or quit.
"""

import argparse
import json
import os

import cv2
import numpy as np

from augment_gen import PICS_CORNERS_FILE, detect_paper_quad, _order_quad

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
GRAB_RADIUS = 18          # display-pixel distance to grab a handle
LABELS = ["TL", "TR", "BR", "BL"]
WINDOW = "edit_corners"


def list_images(pics_dir):
    return sorted(
        f for f in os.listdir(pics_dir)
        if os.path.splitext(f)[1].lower() in IMG_EXTS
    )


def full_frame_quad(w, h):
    return [[0, 0], [w, 0], [w, h], [0, h]]


def detection_to_quad(img, w, h):
    """Run auto-detect and clip result to image bounds. Returns a quad or None."""
    quad = detect_paper_quad(img)
    if quad is None:
        return None
    return [[min(w, max(0.0, float(x))), min(h, max(0.0, float(y)))] for x, y in quad]


def canonicalize(quad, w, h):
    """Re-order TL,TR,BR,BL and clip to image bounds (matches augment_gen)."""
    ordered = _order_quad([list(p) for p in quad])
    return [[max(0, min(w, int(round(x)))), max(0, min(h, int(round(y))))]
            for x, y in ordered]


class CornerEditor:
    def __init__(self, pics_dir, max_dim):
        self.pics_dir = pics_dir
        self.max_dim = max_dim
        self.cache_path = os.path.join(pics_dir, PICS_CORNERS_FILE)

        self.images = list_images(pics_dir)
        if not self.images:
            raise SystemExit(f"No images found in {pics_dir}")

        self.cache = {}
        if os.path.exists(self.cache_path):
            with open(self.cache_path) as f:
                self.cache = json.load(f)

        self.idx = 0
        self.img = None          # current full-res BGR image
        self.h = self.w = 0
        self.scale = 1.0
        self.quad = None         # working copy, full-res float coords [[x,y]*4]
        self.drag = None         # index of handle being dragged, or None
        self.dirty_disk = False  # unsaved changes since last write

        self.load_current()

    # -- image / quad lifecycle -------------------------------------------
    def load_current(self):
        fname = self.images[self.idx]
        self.img = cv2.imread(os.path.join(self.pics_dir, fname))
        if self.img is None:
            raise SystemExit(f"Cannot read {fname}")
        self.h, self.w = self.img.shape[:2]
        self.scale = min(1.0, self.max_dim / max(self.h, self.w))

        quad = self.cache.get(fname)
        if quad is None:
            quad = detection_to_quad(self.img, self.w, self.h) or full_frame_quad(self.w, self.h)
        self.quad = [[float(x), float(y)] for x, y in quad]
        self.drag = None

    def commit(self):
        """Write the working quad back into the in-memory cache (canonical)."""
        fname = self.images[self.idx]
        self.cache[fname] = canonicalize(self.quad, self.w, self.h)
        self.dirty_disk = True

    def save_disk(self):
        self.commit()
        with open(self.cache_path, "w") as f:
            json.dump(self.cache, f, indent=2)
        self.dirty_disk = False
        print(f"Saved {len(self.cache)} entries → {self.cache_path}")

    def goto(self, delta):
        self.commit()
        self.idx = (self.idx + delta) % len(self.images)
        self.load_current()

    # -- coordinate helpers -----------------------------------------------
    def to_disp(self, pt):
        return int(pt[0] * self.scale), int(pt[1] * self.scale)

    def to_full(self, x, y):
        return [min(self.w, max(0, x / self.scale)),
                min(self.h, max(0, y / self.scale))]

    # -- mouse ------------------------------------------------------------
    def on_mouse(self, event, x, y, flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            for i, p in enumerate(self.quad):
                dx, dy = self.to_disp(p)
                if (dx - x) ** 2 + (dy - y) ** 2 <= GRAB_RADIUS ** 2:
                    self.drag = i
                    break
        elif event == cv2.EVENT_MOUSEMOVE and self.drag is not None:
            self.quad[self.drag] = self.to_full(x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.drag = None

    # -- rendering --------------------------------------------------------
    def render(self):
        disp = cv2.resize(self.img, (int(self.w * self.scale), int(self.h * self.scale))) \
            if self.scale < 1.0 else self.img.copy()

        pts = np.array([self.to_disp(p) for p in self.quad], dtype=np.int32)
        cv2.polylines(disp, [pts.reshape(-1, 1, 2)], True, (0, 255, 0), 2)
        for i, (dx, dy) in enumerate(pts):
            active = i == self.drag
            cv2.circle(disp, (dx, dy), 9, (0, 165, 255) if active else (0, 0, 255), -1)
            cv2.putText(disp, LABELS[i], (dx + 12, dy + 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        star = "*" if self.dirty_disk else ""
        header = f"[{self.idx + 1}/{len(self.images)}] {self.images[self.idx]}{star}"
        cv2.putText(disp, header, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 0, 0), 4)
        cv2.putText(disp, header, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 1)
        cv2.imshow(WINDOW, disp)

    # -- main loop --------------------------------------------------------
    def run(self):
        cv2.namedWindow(WINDOW, cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(WINDOW, self.on_mouse)
        print(__doc__.split("Controls:")[1])
        while True:
            self.render()
            key = cv2.waitKey(20) & 0xFF
            if key in (ord("q"), 27):           # q / Esc
                self.save_disk()
                break
            elif key in (ord("n"), 83, 84):     # n / → / ↓
                self.goto(1)
            elif key in (ord("p"), 81, 82):     # p / ← / ↑
                self.goto(-1)
            elif key == ord("s"):
                self.save_disk()
            elif key == ord("r"):
                quad = detection_to_quad(self.img, self.w, self.h)
                if quad is None:
                    print("  auto-detect failed; leaving corners unchanged")
                else:
                    self.quad = quad
            elif key == ord("f"):
                self.quad = [[float(x), float(y)]
                             for x, y in full_frame_quad(self.w, self.h)]
            elif cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                self.save_disk()
                break
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Interactive paper-corner editor.")
    parser.add_argument("--pics-dir", default="images/images_pics")
    parser.add_argument("--max-dim", type=int, default=1100,
                        help="Max display dimension; full-res coords are preserved")
    args = parser.parse_args()
    CornerEditor(args.pics_dir, args.max_dim).run()


if __name__ == "__main__":
    main()
