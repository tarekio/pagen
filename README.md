# Pagen: Arabic Text-Detection Data Generator

Pagen generates synthetic Arabic document images with **word-level polygon annotations** for training and evaluating text-detection (and OCR) models. It produces train and val splits in a single command, with realistic augmentation (scan textures, photo perspective warp, photometric degradation) fused into generation — one final image per document, no intermediate copies.

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

## Project Structure

```
pagen/              Python package (single entry point)
  corpus.py         Word list loading
  fonts.py          Font discovery (skips color fonts)
  llm.py            OpenAI-compatible LLM client (default: Ollama)
  text.py           Arabic normalization, digit conversion, placeholder fill
  render.py         Markdown → PDF → PNG + polygons (in-memory)
  augment.py        Scan / photo-warp / clean augmentation paths
  corners.py        Paper corner cache, interactive editor, overlay export
  visualize.py      Dataset validator and polygon overlay renderer
  templates.py      LLM-based markdown template generation
  pipeline.py       Fused render+augment worker, train/val/eval orchestration
  cli.py            All subcommands + interactive wizard
templates/          Markdown document templates (reused across runs)
fonts/              Arabic .ttf fonts (user-provided)
images/
  images_pics/      Photo backgrounds for perspective warp augmentation
  images_scan/      Scan textures for multiply-blend augmentation
corpus.txt          Word list for random placeholder fill (user-provided)
```

## Requirements

### System dependencies (Linux/Debian)

WeasyPrint needs Pango, Cairo, and friends. PyMuPDF bundles its own MuPDF — no extra system packages needed for rasterization.

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

Drop Arabic `.ttf` files into `fonts/`. A good source is [Google Fonts](https://fonts.google.com/?subset=arabic). Color fonts (COLR/SVG) are skipped automatically.

### Corpus

Create `corpus.txt` with one or more Arabic words per line (space-separated is fine). Without it, a tiny built-in fallback is used. You can also point to a directory of word files via `--corpus`.

## Quickstart

```bash
# Interactive wizard — prompts for counts, output dir, mode
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

### `pagen dataset` — generate a detection dataset

```bash
pagen dataset --train N --val N [options]
```

| Flag | Default | Description |
|---|---|---|
| `--train N` | — | Number of training documents |
| `--val N` | — | Number of validation documents |
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
| `--scan-prob` | 0.45 | Fraction composited onto a scan texture |
| `--pics-prob` | 0.45 | Fraction perspective-warped onto a background photo |
| `--scan-dir` | `images/images_scan` | Scan texture images |
| `--pics-dir` | `images/images_pics` | Background photo images |

**LLM content fill** (off by default — uses random corpus words):

| Flag | Default | Description |
|---|---|---|
| `--llm` | off | Enable LLM-based placeholder fill |
| `--llm-base-url` | `http://localhost:11434/v1` | OpenAI-compatible API endpoint |
| `--llm-model` | `llama3` | Model name |
| `--api-key-env` | `OPENAI_API_KEY` | Env var holding the API key (never a CLI flag) |

Re-running into an existing output directory **appends** — new IDs continue after existing ones and `labels.json` is preserved.

**Examples:**

```bash
# 1000 augmented training docs + 200 val, random fill
pagen dataset --train 1000 --val 200 -o data/

# Clean (no augmentation), save txt and pdf alongside images
pagen dataset --train 50 --val 10 --pdf-only --keep-txt --keep-pdf -o data/

# LLM fill via Ollama (must be running: ollama serve)
pagen dataset --train 100 --val 20 --llm --llm-model llama3

# LLM fill via any OpenAI-compatible provider
OPENAI_API_KEY=sk-... pagen dataset --train 100 --val 20 \
  --llm --llm-base-url https://api.openai.com/v1 --llm-model gpt-4o-mini
```

---

### `pagen eval` — generate eval images (no polygon annotations)

Produces a PNG + plain-text ground truth per document. No `labels.json`.

```bash
pagen eval -c 50 -o data/eval
```

---

### `pagen templates` — generate markdown templates via LLM

Templates are generated once and reused across many dataset runs. This command is **never** invoked automatically by the dataset pipeline.

```bash
# Generate 10 templates from the built-in document-type pool
pagen templates --random 10 --llm --llm-model llama3

# Generate specific types
pagen templates "عقد عمل" "خطاب توصية" --llm

# From a file
pagen templates --file my_types.txt --llm
```

---

### `pagen corners` — manage paper corner cache

The photo background augmentation path needs to know where the paper is in each photo. Corners are detected automatically the first time and cached in `paper_corners.json`. Each entry records its provenance (`"source": "auto"` or `"user"`). The cache is reconciled **append-only**: only newly-added images are auto-detected and saved — existing entries are never re-detected or overwritten, so corners you fix in the editor (`"source": "user"`) are safe across dataset runs.

```bash
# Build / refresh the cache (runs automatically during augmented dataset gen)
pagen corners --pics-dir images/images_pics

# Launch interactive editor to fix bad auto-detections
pagen corners --pics-dir images/images_pics --edit

# Export overlay images to inspect detections
pagen corners --pics-dir images/images_pics --visualize --out corners_vis/
```

**Editor controls:** drag handle = move corner | `n`/`→` = next | `p`/`←` = prev | `r` = re-detect | `f` = full-frame | `s` = save | `q`/Esc = save + quit

---

### `pagen visualize` — validate and overlay a dataset

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

Ollama requires no real key — the client sends a dummy value it ignores.
