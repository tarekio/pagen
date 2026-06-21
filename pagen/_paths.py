"""Default locations for bundled example resources.

These are CWD-relative so `pagen` runs from the repo root out of the box.
Users override per-run via --fonts-dir, --templates-dir, --scene-dir, etc.
"""

import os

RESOURCES_DIR = "resources"

FONTS_DIR = os.path.join(RESOURCES_DIR, "fonts")
CORPORA_DIR = os.path.join(RESOURCES_DIR, "corpora")
TEMPLATES_DIR = os.path.join(RESOURCES_DIR, "templates")
SCENE_DIR = os.path.join(RESOURCES_DIR, "images", "scene")        # photo backgrounds
TEXTURES_DIR = os.path.join(RESOURCES_DIR, "images", "textures")  # scan textures

DEFAULT_FONT = os.path.join(FONTS_DIR, "Amiri-Regular.ttf")
