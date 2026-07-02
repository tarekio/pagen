# Pagen: Arabic Text-Detection Data Generator

Pagen generates synthetic realistic-looking and diverse document images with **word-level polygon annotations** for training and evaluating text-detection (and OCR) models. 

It produces train and val splits in a single command, with realistic augmentation (scan textures, photo perspective warp, photometric degradation) fused into generation: one final image per document, no intermediate copies.

The output follows the [doctr](https://github.com/mindee/doctr) detection-dataset layout:

```json
{
  "1.png": {
    "img_dimensions": [width, height],
    "img_hash": "<sha256 of the png>",
    "polygons": [[[x1, y1], [x2, y1], [x2, y2], [x1, y2]], ...],
    "labels": ["الكلمة", "الأولى", ...]
  }
}
```

Each polygon is a 4-point quadrilateral (TL, TR, BR, BL) wrapping a single word. Rendering uses WeasyPrint (Markdown → HTML → PDF); word labels come from WeasyPrint's layout tree so Arabic shaping and Eastern-Arabic digits are always correct. PyMuPDF rasterizes the PDF and supplies tight per-glyph geometry so polygons fit the visible ink rather than the loose em-box.

## Requirements

### System dependencies (Linux/Debian)

WeasyPrint needs Pango, Cairo, and friends. PyMuPDF bundles its own MuPDF; no extra system packages are needed for rasterization.

```bash
sudo apt-get install build-essential python3-dev python3-cffi \
  libcairo2 libpango-1.0-0 libpangocairo-1.0-0 libgdk-pixbuf2.0-0 \
  libffi-dev shared-mime-info
```

### Python dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
# or without editable install:
pip install -r requirements.txt
```

Optional extras:

```bash
pip install python-dotenv          # auto-load .env for API keys
pip install arabic-reshaper matplotlib  # pagen visualize --show
```

### Fonts

Drop Arabic `.ttf` files into `resources/fonts/` (or pass `--fonts-dir`). A good source is [Google Fonts](https://fonts.google.com/?subset=arabic). Color fonts (COLR/SVG) are skipped automatically.

### Corpus

Add word files to `resources/corpora/` (one or more Arabic words per line; space-separated is fine), or point `--corpus` at a file or directory. Without any corpus, a tiny built-in fallback is used.

### Images

Augmentation draws on two folders of your own photos. [Unsplash](https://unsplash.com) is a good free source for both.

- **`resources/images/scene/`** (`--scene-dir`): photos of a **blank sheet of paper lying in a real scene** (on a desk, table, floor, with surrounding clutter). The rendered document is perspective-warped onto the paper region and composited into the photo, so the model sees documents at realistic angles against busy backgrounds. The paper's four corners are detected automatically and cached in `paper_corners.json` (editable via `pagen corners --edit`), so pick photos where the sheet is clearly visible against its surroundings. Search Unsplash for *blank paper*, *paper on desk*, *empty document*.
- **`resources/images/textures/`** (`--textures-dir`): flat **paper / scan textures** (plain paper grain, aged or stained paper, photocopier noise) used as full-frame overlays. The rendered document is multiply-blended onto the texture to mimic a scanned or photocopied page, with no perspective, just surface character. Search Unsplash for *paper texture*, *old paper*, *parchment*.

The `--clean-prob` / `--textures-prob` / `--scene-prob` flags control how often each path (clean, texture overlay, scene warp) is chosen. If a folder is empty, its path is simply skipped.

## Quickstart

```bash
# Interactive wizard: prompts for counts, output dir, mode
pagen

# Or specify everything:
pagen dataset --train 500 --val 100 -o data/
```

The result loads directly with doctr:

```python
from doctr.datasets import DetectionDataset
ds = DetectionDataset(img_folder="data/train/images", label_path="data/train/labels.json")
```

## Commands

### `pagen dataset`: generate a detection dataset

```bash
pagen dataset --train N --val N [options]
```

| Flag | Default | Description |
|---|---|---|
| `--train N` | `0` | Number of training documents (omit to skip the split; at least one of train/val must be > 0) |
| `--val N` | `0` | Number of validation documents (omit to skip the split) |
| `-o DIR` | `output` | Root output directory; splits go in `DIR/train` and `DIR/val` |
| `--pdf-only` | off | Skip augmentation; produce clean rendered images |
| `--keep-txt` | off | Save plain-text word labels per page (`.txt`) |
| `--keep-pdf` | off | Save the intermediate PDF per document |
| `--dpi` | 150 | Raster resolution |
| `--workers` | all CPUs | Parallel worker processes |
| `--seed` | 42 | RNG seed |
| `--templates-dir` | `templates` | Directory of `.md` templates |
| `--corpus` | auto | Corpus file or directory |
| `--fonts-dir` | `fonts` | Directory of `.ttf` fonts |

**Augmentation flags** (ignored with `--pdf-only`):

| Flag | Default | Description |
|---|---|---|
| `--clean-prob` | 0.10 | Fraction of images kept clean |
| `--textures-prob` | 0.45 | Fraction composited onto a scan texture |
| `--scene-prob` | 0.45 | Fraction perspective-warped onto a background photo |
| `--textures-dir` | `resources/images/textures` | Scan texture images |
| `--scene-dir` | `resources/images/scene` | Background photo images |

**LLM content fill** (off by default; uses random corpus words):

| Flag | Default | Description |
|---|---|---|
| `--llm` | off | Enable LLM-based placeholder fill |
| `--llm-base-url` | `http://localhost:11434/v1` | OpenAI-compatible API endpoint |
| `--llm-model` | `qwen2.5:7b` | Model name |
| `--api-key-env` | `OPENAI_API_KEY` | Env var holding the API key (never a CLI flag) |
| `--fill-variants` | `10` | LLM fills pre-generated per template, then sampled by workers |

With `--llm`, content is generated **once per template** (`--fill-variants` variants
each) in a single pass before rendering, and workers sample from that pool. LLM
calls therefore scale with `templates * fill-variants`, not with the number of
images. If the backend is unreachable, generation falls back to random corpus fill.

Re-running into an existing output directory **appends**: new IDs continue after existing ones and `labels.json` is preserved.

**Examples:**

```bash
# 1000 augmented training docs + 200 val, random fill
pagen dataset --train 1000 --val 200 -o data/

# Clean (no augmentation), save txt and pdf alongside images
pagen dataset --train 50 --val 10 --pdf-only --keep-txt --keep-pdf -o data/

# LLM fill via Ollama (must be running: ollama serve)
pagen dataset --train 100 --val 20 --llm --llm-model qwen2.5:7b

# LLM fill via any OpenAI-compatible provider
OPENAI_API_KEY=sk-... pagen dataset --train 100 --val 20 \
  --llm --llm-base-url https://api.openai.com/v1 --llm-model gpt-4o-mini
```

---

### `pagen eval`: generate eval images (no polygon annotations)

Produces a PNG + plain-text ground truth per document. No `labels.json`.

```bash
pagen eval -c 50 -o data/eval
```

---

### `pagen templates`: generate markdown templates via LLM

Templates are generated once and reused across many dataset runs. This command is **never** invoked automatically by the dataset pipeline.

```bash
# Generate 10 templates from the built-in document-type pool
pagen templates --random 10 --llm --llm-model qwen2.5:7b

# Generate specific types
pagen templates "عقد عمل" "خطاب توصية" --llm

# From a file
pagen templates --file my_types.txt --llm
```

---

### `pagen corners`: manage paper corner cache

The photo background augmentation path needs to know where the paper is in each photo. Corners are detected automatically the first time and cached in `paper_corners.json`. The automatic detection is not perfect, so you can inspect and fix the cache with an interactive editor. 

Each entry records its provenance (`"source": "auto"` or `"user"`). The cache is reconciled **append-only**: only newly-added images are auto-detected and saved; existing entries are never re-detected or overwritten, so corners you fix in the editor (`"source": "user"`) are safe across dataset runs.

```bash
# Build / refresh the cache (runs automatically during augmented dataset gen)
pagen corners --scene-dir resources/images/scene

# Launch interactive editor to fix bad auto-detections
pagen corners --scene-dir resources/images/scene --edit

# Export overlay images to inspect detections
pagen corners --scene-dir resources/images/scene --visualize --out corners_vis/
```

**Editor controls:** drag handle = move corner | `n`/`→` = next | `p`/`←` = prev | `r` = re-detect | `f` = full-frame | `s` = save | `q`/Esc = save + quit

---

### `pagen visualize`: validate and overlay a dataset

```bash
pagen visualize data/train --max 20 --save data/train/debug_vis/
pagen visualize data/train --show   # requires matplotlib
```

Reports out-of-bounds polygons, label/polygon count mismatches, and empty labels.

---

## API key security

API keys cannot be passed as CLI flags. Set the environment variable named by `--api-key-env` (default `OPENAI_API_KEY`):

```bash
export OPENAI_API_KEY=sk-...
# or put it in a .env file (auto-loaded if python-dotenv is installed):
echo "OPENAI_API_KEY=sk-..." > .env
```

Ollama requires no real key; the client sends a dummy value it ignores.
