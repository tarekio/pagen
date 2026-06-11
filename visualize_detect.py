"""Debug visualizer for the detection dataset produced by detect_gen.py.

Loads a doctr-format dataset (``<dir>/labels.json`` + ``<dir>/images/``), validates
each entry (size match, in-bounds polygons, label/polygon parity) and draws the word
polygons with their (reshaped, RTL-correct) Arabic labels.

Use it two ways:

  # CLI: write annotated overlays to <dir>/debug_vis/
  python visualize_detect.py output/ --max 10

  # Notebook: draw inline with matplotlib
  from visualize_detect import visualize_dataset
  visualize_dataset("output/", show=True, max_images=5)

Extra (debug-only) dependencies for nice Arabic labels:
  pip install arabic-reshaper python-bidi   # and matplotlib for --show
"""

import argparse
import json
import os
import random

from PIL import Image, ImageDraw, ImageFont

# Optional: correct Arabic shaping + RTL ordering for the drawn label text.
try:
    import arabic_reshaper

    def shape_arabic(text):
        return arabic_reshaper.reshape(text)

except ImportError:  # labels still draw, just disconnected / left-to-right

    def shape_arabic(text):
        return text


DEFAULT_FONT = os.path.join(os.path.dirname(__file__), "fonts", "Amiri-Regular.ttf")


def _load_font(font_path, size):
    try:
        return ImageFont.truetype(font_path, size)
    except OSError:
        return ImageFont.load_default()


def validate_entry(img_name, info, real_size):
    """Return a list of human-readable problems with one dataset entry (empty == OK)."""
    problems = []
    polygons = info.get("polygons", [])
    labels = info.get("labels", [])
    size = info.get("img_dimensions", [0, 0])

    if list(size) != list(real_size):
        problems.append(f"img_dimensions {size} != actual {list(real_size)}")

    w, h = size if len(size) == 2 else (0, 0)
    oob = 0
    for poly in polygons:
        for x, y in poly:
            if x < 0 or x > w or y < 0 or y > h:
                oob += 1
    if oob:
        problems.append(f"{oob} polygon point(s) out of bounds for {size}")

    if len(polygons) != len(labels):
        problems.append(f"{len(polygons)} polygons but {len(labels)} labels")
    if not labels:
        problems.append("no labels")
    elif any(not str(lbl).strip() for lbl in labels):
        problems.append("contains empty label(s)")

    return problems


def annotate_image(image, polygons, labels, font, show_labels=True):
    """Draw polygons (green) and labels (red, above each box) onto a PIL image copy."""
    out = image.convert("RGB")
    draw = ImageDraw.Draw(out)
    for i, poly in enumerate(polygons):
        draw.polygon([tuple(p) for p in poly], outline=(0, 255, 0), width=2)
        if show_labels and i < len(labels):
            x, y = poly[0]
            draw.text((x, y - 28), shape_arabic(str(labels[i])), font=font, fill=(255, 0, 0))
    return out


def visualize_dataset(
    path,
    show_labels=True,
    max_images=10,
    debug=False,
    show=False,
    save_dir=None,
    seed=None,
    font_path=DEFAULT_FONT,
    font_size=24,
):
    """Validate and render a detection dataset at ``path`` (a directory holding
    ``labels.json`` and ``images/``). Saves annotated overlays to ``save_dir`` (default
    ``<path>/debug_vis``) and/or shows them inline when ``show=True``."""
    labels_path = os.path.join(path, "labels.json")
    images_dir = os.path.join(path, "images")
    with open(labels_path, "r", encoding="utf-8") as f:
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

        annotated = annotate_image(
            image, info.get("polygons", []), info.get("labels", []), font, show_labels
        )
        if save_dir:
            annotated.save(os.path.join(save_dir, img_name))
        if show:
            import matplotlib.pyplot as plt

            plt.figure(figsize=(12, 8))
            plt.imshow(annotated)
            plt.title(img_name)
            plt.axis("off")
            plt.show()

        shown += 1
        if shown >= max_images:
            break

    where = f"saved to {save_dir}" if save_dir else "shown"
    print(f"Visualized {shown} image(s) ({where}); {total_problems} problem(s) found.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize/validate a detect_gen.py detection dataset."
    )
    parser.add_argument("path", help="Dataset directory (contains labels.json and images/)")
    parser.add_argument("--max", type=int, default=1, help="Max images to render (default: 10)")
    parser.add_argument("--save", default=None, help="Directory for overlays (default: <path>/debug_vis)")
    parser.add_argument("--show", action="store_true", help="Show inline with matplotlib instead of saving")
    parser.add_argument("--no-labels", action="store_true", help="Draw polygons only, no label text")
    parser.add_argument("--seed", type=int, default=None, help="Shuffle seed for reproducible sampling")
    parser.add_argument("--font", default=DEFAULT_FONT, help="TTF font for label text")
    parser.add_argument("--debug", action="store_true", help="Print a line per validated image")
    args = parser.parse_args()

    visualize_dataset(
        args.path,
        show_labels=not args.no_labels,
        max_images=args.max,
        debug=args.debug,
        show=args.show,
        save_dir=args.save,
        seed=args.seed,
        font_path=args.font,
    )
