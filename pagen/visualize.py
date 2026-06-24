"""Detection dataset validator and polygon/label overlay renderer.

Folds in visualize_detect.py.  The key function ``visualize_dataset()`` is
importable directly:

    from pagen.visualize import visualize_dataset
    visualize_dataset("output/train", show=True, max_images=5)

Optional display deps: arabic-reshaper, matplotlib.
"""

from __future__ import annotations

import json
import os
import random

from PIL import Image, ImageDraw, ImageFont

try:
    import arabic_reshaper
    def _shape(text: str) -> str:
        return arabic_reshaper.reshape(text)
except ImportError:
    def _shape(text: str) -> str:
        return text

from pagen._paths import DEFAULT_FONT


def _load_font(font_path: str | None, size: int):
    # font_path may be None when no usable font was found (e.g. the bundled
    # fonts are gitignored and absent on a fresh checkout).  Fall back to the
    # built-in default rather than letting ImageFont.truetype(None) raise.
    if not font_path:
        return ImageFont.load_default()
    try:
        return ImageFont.truetype(font_path, size)
    except OSError:
        return ImageFont.load_default()


def validate_entry(img_name: str, info: dict, real_size: tuple) -> list[str]:
    """Return a list of human-readable problems with one dataset entry (empty == OK)."""
    problems = []
    polygons = info.get("polygons", [])
    labels = info.get("labels", [])
    size = info.get("img_dimensions", [0, 0])

    if list(size) != list(real_size):
        problems.append(f"img_dimensions {size} != actual {list(real_size)}")

    w, h = size if len(size) == 2 else (0, 0)
    oob = sum(
        1 for poly in polygons for x, y in poly
        if x < 0 or x > w or y < 0 or y > h
    )
    if oob:
        problems.append(f"{oob} polygon point(s) out of bounds for {size}")

    if len(polygons) != len(labels):
        problems.append(f"{len(polygons)} polygons but {len(labels)} labels")
    if not labels:
        problems.append("no labels")
    elif any(not str(lbl).strip() for lbl in labels):
        problems.append("contains empty label(s)")

    return problems


def annotate_image(image: Image.Image, polygons: list, labels: list, font, show_labels: bool = True) -> Image.Image:
    """Draw green polygons and red labels onto a PIL image copy."""
    out = image.convert("RGB")
    draw = ImageDraw.Draw(out)
    for i, poly in enumerate(polygons):
        draw.polygon([tuple(p) for p in poly], outline=(0, 255, 0), width=2)
        if show_labels and i < len(labels):
            x, y = poly[0]
            draw.text((x, y - 28), _shape(str(labels[i])), font=font, fill=(255, 0, 0))
    return out


def visualize_dataset(
    path: str,
    show_labels: bool = True,
    max_images: int = 10,
    debug: bool = False,
    show: bool = False,
    save_dir: str | None = None,
    seed: int | None = None,
    font_path: str = DEFAULT_FONT,
    font_size: int = 24,
) -> None:
    """Validate and render a detection dataset at ``path``.

    ``path`` must contain ``labels.json`` and ``images/``.  Saves annotated
    overlays to ``save_dir`` (default ``<path>/debug_vis``) and/or shows inline
    when ``show=True``.
    """
    labels_path = os.path.join(path, "labels.json")
    images_dir = os.path.join(path, "images")
    with open(labels_path, encoding="utf-8") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} entries from {labels_path}")

    if save_dir is None and not show:
        save_dir = os.path.join(path, "debug_vis")
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    font = _load_font(font_path, font_size)
    items = list(data.items())
    random.Random(seed).shuffle(items)

    total_problems = 0
    shown = 0
    for img_name, info in items:
        image_path = os.path.join(images_dir, img_name)
        if not os.path.exists(image_path):
            print(f"  [skip] {img_name}: image not found at {image_path}")
            continue

        image = Image.open(image_path)
        problems = validate_entry(img_name, info, image.size)
        total_problems += len(problems)
        if problems:
            print(f"  [warn] {img_name}: " + "; ".join(problems))
        elif debug:
            print(f"  [ok]   {img_name}: {len(info.get('polygons', []))} polygons")

        annotated = annotate_image(image, info.get("polygons", []), info.get("labels", []), font, show_labels)
        if save_dir:
            annotated.save(os.path.join(save_dir, img_name))
        if show:
            try:
                import matplotlib.pyplot as plt
                plt.figure(figsize=(12, 8))
                plt.imshow(annotated)
                plt.title(img_name)
                plt.axis("off")
                plt.show()
            except ImportError:
                print("  [warn] matplotlib not installed; cannot show inline. Install with: pip install matplotlib")

        shown += 1
        if shown >= max_images:
            break

    where = f"saved to {save_dir}" if save_dir else "shown"
    print(f"Visualized {shown} image(s) ({where}); {total_problems} problem(s) found.")
