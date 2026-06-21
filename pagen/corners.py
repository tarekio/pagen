"""Paper corner cache management, interactive editor, and overlay export.

Folds in edit_corners.py and visualize_corners.py.

Subcommand surface (via cli.py):
  pagen corners [--pics-dir DIR]           # rebuild/refresh the cache
  pagen corners --edit [--max-dim N]       # launch interactive editor
  pagen corners --visualize [--out DIR]    # write overlay images
"""

from __future__ import annotations

import os

import cv2
import numpy as np

from pagen.augment import (
    PICS_CORNERS_FILE,
    detect_paper_quad,
    ensure_corners_cache,
    load_corners_cache,
    write_corners_cache,
    quad_of,
    _make_entry,
    _order_quad,
)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
GRAB_RADIUS = 18
LABELS = ["TL", "TR", "BR", "BL"]
WINDOW = "edit_corners"


# ---------------------------------------------------------------------------
# Helpers shared by editor and visualizer
# ---------------------------------------------------------------------------

def _list_images(pics_dir: str) -> list[str]:
    return sorted(f for f in os.listdir(pics_dir) if os.path.splitext(f)[1].lower() in IMG_EXTS)


def _full_frame_quad(w, h):
    return [[0, 0], [w, 0], [w, h], [0, h]]


def _canonicalize(quad, w, h):
    ordered = _order_quad([list(p) for p in quad])
    return [[max(0, min(w, int(round(x)))), max(0, min(h, int(round(y))))] for x, y in ordered]


def _draw_quad(img, quad, color=(0, 255, 0), thickness=3):
    pts = np.array(quad, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(img, [pts], isClosed=True, color=color, thickness=thickness)
    for i, (x, y) in enumerate(quad):
        cv2.circle(img, (int(x), int(y)), 8, (0, 0, 255), -1)
        cv2.putText(img, LABELS[i], (int(x) + 10, int(y) + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    return img


# ---------------------------------------------------------------------------
# Interactive corner editor
# ---------------------------------------------------------------------------

class _CornerEditor:
    def __init__(self, pics_dir: str, max_dim: int):
        self.pics_dir = pics_dir
        self.max_dim = max_dim
        self.cache_path = os.path.join(pics_dir, PICS_CORNERS_FILE)

        self.images = _list_images(pics_dir)
        if not self.images:
            raise SystemExit(f"No images found in {pics_dir}")

        self.cache: dict = load_corners_cache(pics_dir)

        self.idx = 0
        self.img = None
        self.h = self.w = 0
        self.scale = 1.0
        self.quad = None
        self.loaded_quad = None   # canonical quad as loaded, to detect real edits
        self.drag = None
        self.dirty_disk = False
        self._load_current()

    def _load_current(self):
        fname = self.images[self.idx]
        self.img = cv2.imread(os.path.join(self.pics_dir, fname))
        if self.img is None:
            raise SystemExit(f"Cannot read {fname}")
        self.h, self.w = self.img.shape[:2]
        self.scale = min(1.0, self.max_dim / max(self.h, self.w))
        entry = self.cache.get(fname)
        if entry is None:
            detected = detect_paper_quad(self.img)
            quad = (detected if detected else _full_frame_quad(self.w, self.h))
        else:
            quad = quad_of(entry)
        self.quad = [[float(x), float(y)] for x, y in quad]
        self.loaded_quad = _canonicalize(self.quad, self.w, self.h)
        self.drag = None

    def _commit(self):
        """Write the current quad into the in-memory cache.

        Provenance is only bumped to 'user' when the quad actually changed from
        what was loaded, so merely navigating past an image never reclassifies
        an auto entry as user-edited.
        """
        fname = self.images[self.idx]
        canon = _canonicalize(self.quad, self.w, self.h)
        prior = self.cache.get(fname)
        if canon == self.loaded_quad and prior is not None:
            source = prior.get("source", "auto")   # unchanged → keep provenance
        else:
            source = "user"                          # new or edited → user-owned
        self.cache[fname] = _make_entry(canon, source)
        self.dirty_disk = True

    def _save_disk(self):
        self._commit()
        write_corners_cache(self.pics_dir, self.cache)
        self.dirty_disk = False
        print(f"Saved {len(self.cache)} entries → {self.cache_path}")

    def _goto(self, delta):
        self._commit()
        self.idx = (self.idx + delta) % len(self.images)
        self._load_current()

    def _to_disp(self, pt):
        return int(pt[0] * self.scale), int(pt[1] * self.scale)

    def _to_full(self, x, y):
        return [min(self.w, max(0, x / self.scale)), min(self.h, max(0, y / self.scale))]

    def _on_mouse(self, event, x, y, flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            for i, p in enumerate(self.quad):
                dx, dy = self._to_disp(p)
                if (dx - x) ** 2 + (dy - y) ** 2 <= GRAB_RADIUS ** 2:
                    self.drag = i
                    break
        elif event == cv2.EVENT_MOUSEMOVE and self.drag is not None:
            self.quad[self.drag] = self._to_full(x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.drag = None

    def _render(self):
        if self.scale < 1.0:
            disp = cv2.resize(self.img, (int(self.w * self.scale), int(self.h * self.scale)))
        else:
            disp = self.img.copy()
        pts = np.array([self._to_disp(p) for p in self.quad], dtype=np.int32)
        cv2.polylines(disp, [pts.reshape(-1, 1, 2)], True, (0, 255, 0), 2)
        for i, (dx, dy) in enumerate(pts):
            active = i == self.drag
            cv2.circle(disp, (dx, dy), 9, (0, 165, 255) if active else (0, 0, 255), -1)
            cv2.putText(disp, LABELS[i], (dx + 12, dy + 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        star = "*" if self.dirty_disk else ""
        header = f"[{self.idx + 1}/{len(self.images)}] {self.images[self.idx]}{star}"
        cv2.putText(disp, header, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
        cv2.putText(disp, header, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        cv2.imshow(WINDOW, disp)

    def run(self):
        """Launch the interactive editor loop."""
        print("Controls: drag handle=move corner | n/→=next | p/←=prev | r=redetect | f=full-frame | s=save | q/Esc=save+quit")
        cv2.namedWindow(WINDOW, cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(WINDOW, self._on_mouse)
        while True:
            self._render()
            key = cv2.waitKey(20) & 0xFF
            if key in (ord("q"), 27):
                self._save_disk()
                break
            elif key in (ord("n"), 83, 84):
                self._goto(1)
            elif key in (ord("p"), 81, 82):
                self._goto(-1)
            elif key == ord("s"):
                self._save_disk()
            elif key == ord("r"):
                detected = detect_paper_quad(self.img)
                if detected is None:
                    print("  auto-detect failed; leaving corners unchanged")
                else:
                    self.quad = [[float(x), float(y)] for x, y in detected]
            elif key == ord("f"):
                self.quad = [[float(x), float(y)] for x, y in _full_frame_quad(self.w, self.h)]
            elif cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                self._save_disk()
                break
        cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

def build_cache(pics_dir: str) -> dict:
    """Detect and persist paper corners for all images in pics_dir."""
    return ensure_corners_cache(pics_dir)


def launch_editor(pics_dir: str, max_dim: int = 1100):
    """Launch the interactive cv2 corner editor."""
    _CornerEditor(pics_dir, max_dim).run()


def export_overlays(pics_dir: str, out_dir: str):
    """Write corner-overlay images (one per photo) to out_dir."""
    cache_path = os.path.join(pics_dir, PICS_CORNERS_FILE)
    if not os.path.exists(cache_path):
        print(f"ERROR: {cache_path} not found. Run `pagen corners` first.")
        return

    cache = load_corners_cache(pics_dir)

    os.makedirs(out_dir, exist_ok=True)
    for fname, entry in sorted(cache.items()):
        quad = quad_of(entry)
        fpath = os.path.join(pics_dir, fname)
        img = cv2.imread(fpath)
        if img is None:
            print(f"  [skip] {fname}: cannot read image")
            continue
        h, w = img.shape[:2]
        max_dim = 1200
        scale = min(1.0, max_dim / max(h, w))
        if scale < 1.0:
            disp = cv2.resize(img, (int(w * scale), int(h * scale)))
            quad_scaled = [[x * scale, y * scale] for x, y in quad]
        else:
            disp = img.copy()
            quad_scaled = quad
        _draw_quad(disp, quad_scaled)
        out_path = os.path.join(out_dir, fname.rsplit(".", 1)[0] + ".jpg")
        cv2.imwrite(out_path, disp)
        print(f"  {fname}: {quad} → {out_path}")
    print(f"Saved {len(cache)} overlays to {out_dir}/")
