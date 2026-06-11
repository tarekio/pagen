# Pagen: Arabic Text-Detection Data Generator

Pagen is a simple Python script that generates single-page synthetic Arabic documents using Markdown templates, producing **word-level polygon annotations** for training and evaluating text-detection (and OCR) models.

It supports random generation from a word list or context-aware completion using local LLMs via Ollama. It properly maps numbers to Eastern Arabic numerals and renders the files as right-to-left.

The output directory follows the [doctr](https://github.com/mindee/doctr) detection-dataset layout: images go in an `images/` subfolder and the annotations are written to a single **`labels.json`** at the root (alongside a plain-text **ground truth** `.txt` per document). Each entry of `labels.json` is keyed by the image's bare filename, in the doctr detection format:

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

Each polygon is a 4-point quadrilateral (top-left, top-right, bottom-right, bottom-left) wrapping a single word, in image pixel coordinates, and `labels[i]` is the text of `polygons[i]` (parallel lists). Rendering is done with WeasyPrint (Markdown → HTML → PDF). The text and word grouping come from **WeasyPrint's layout tree** (each word is wrapped in a span), so the labels are the source text — Arabic shaping and Eastern-Arabic digits stay correct, unlike text extracted from a PDF. **PyMuPDF** rasterizes that same PDF to the PNG and also supplies the tight per-glyph geometry: for each word, the glyph boxes inside it are unioned (so descenders and detached dots are fully enclosed) and then shrunk to the actual rendered ink, so the polygon fits the visible glyphs snugly regardless of the font's em-box metrics.

## Project Structure
- `detect_gen.py`: The detection-dataset generator (PNG + TXT + polygon `labels.json`), described below.
- `eval_gen.py`: A simpler evaluation generator that outputs only an image (PDF/PNG) and a TXT ground truth per document, with no polygon annotations.
- `templates/`: Directory containing Markdown template files.
- `fonts/`: (User provided) Directory containing custom Arabic `.ttf` font files used when rendering. You will need to add your own `.ttf` fonts here. A good source is [Google Fonts](https://fonts.google.com/?subset=arabic).
- `corpus.txt`: (User provided) Create your own text file containing lexicon words for random generation, or bypass this by using the `--ollama` flag to generate content.

## Requirements

WeasyPrint needs a few system libraries (Pango, Cairo, etc.). `detect_gen.py` rasterizes via PyMuPDF, which bundles its own MuPDF (no system packages). `eval_gen.py` rasterizes via `pdf2image`, which needs Poppler (`poppler-utils`).

### System Dependencies

**For Linux (Ubuntu/Debian):**
```bash
sudo apt-get update
# Dependencies for weasyprint (pango, cairo, etc.); poppler-utils is for eval_gen.py (pdf2image)
sudo apt-get install build-essential python3-dev python3-pip python3-setuptools python3-wheel python3-cffi libcairo2 libpango-1.0-0 libpangocairo-1.0-0 libgdk-pixbuf2.0-0 libffi-dev shared-mime-info poppler-utils
```

### Python Dependencies

Install the required Python packages from the `requirements.txt`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

*(Note: `ollama` is optional but recommended if you want realistic document generation using local LLMs. `weasyprint` and `pymupdf` are required — WeasyPrint renders the PDF, PyMuPDF rasterizes it to PNG and extracts the word polygons).*

## Generation Instructions

Run `detect_gen.py` to generate documents. The script processes a random template, populates the content placeholders, writes a PNG image to `images/` and a TXT ground truth per document, and appends the polygon annotations to `labels.json` in the output directory.

```bash
python detect_gen.py [options]
```

### Command Line Arguments

- `-o`, `--output` : The output directory for the dataset (defaults to `output`). Images are written to `output/images/` and `output/labels.json` is written/updated here.
- `-c`, `--count` : Number of documents to generate (defaults to `1`).
- `--dpi` : Raster resolution for the PNGs (defaults to `150`).
- `--keep-pdf` : Also save the intermediate `.pdf` for each document (off by default).
- `--ollama` : Use Ollama to generate context-appropriate text instead of random words (requires the `ollama` package).
- `--ollama-model` : Specify an Ollama model to use, for example `llama3` (defaults to `llama3`).

Re-running into an existing output directory **appends** to `labels.json` (new images are numbered after the existing ones in `images/`) rather than overwriting it.

The result is ready to load with doctr's `DetectionDataset(img_folder="output/images", label_path="output/labels.json")`.

### Examples

**Generate 5 random documents into the `output` directory:**
```bash
python detect_gen.py -c 5
```

**Generate 10 documents into a specific `data/` directory using Ollama (Llama 3):**
```bash
python detect_gen.py -c 10 -o data --ollama --ollama-model llama3
```

*(If you use Ollama, ensure the Ollama server is running locally and the model `llama3` is pulled: `ollama run llama3`)*
