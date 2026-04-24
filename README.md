# Pagen: Arabic Document Generator

Pagen is a simple Python script that generates single-page synthetic Arabic documents (in TXT, PDF, and PNG formats) using Markdown templates for training and evaluating OCR models. 

It supports random generation from a word list or context-aware completion using local LLMs via Ollama. It properly maps numbers to Eastern Arabic numerals and renders the files as right-to-left.

The script outputs a plain text file (ground truth), a PDF file, and a PNG image file for each one-page document generated.

## Project Structure
- `generate_doc.py`: The main generation script.
- `templates/`: Directory containing Markdown template files.
- `fonts/`: (User provided) Directory containing custom Arabic `.ttf` font files used when generating the PDFs. You will need to add your own `.ttf` fonts here. A good source is [Google Fonts](https://fonts.google.com/?subset=arabic).
- `corpus.txt`: (User provided) Create your own text file containing lexicon words for random generation, or bypass this by using the `--ollama` flag to generate content.

## Requirements

The project generates PDFs and converts them to PNG images natively, so you'll need a few system dependencies along with Python packages.

### System Dependencies

**For Linux (Ubuntu/Debian):**
```bash
sudo apt-get update
# Dependencies for weasyprint (pango, cairo, etc.) and pdf2image (poppler)
sudo apt-get install build-essential python3-dev python3-pip python3-setuptools python3-wheel python3-cffi libcairo2 libpango-1.0-0 libpangocairo-1.0-0 libgdk-pixbuf2.0-0 libffi-dev shared-mime-info poppler-utils
```

### Python Dependencies

Install the required Python packages from the `requirements.txt`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

*(Note: `ollama` is optional but recommended if you want realistic document generation using local LLMs. `weasyprint` and `pdf2image` are required for PDF and PNG creation).*

## Generation Instructions

Run `generate_doc.py` to generate documents. The script processes a random template, populates the content placeholders, and outputs a TXT file (plain text ground truth), a PDF file, and a PNG image file.

```bash
python generate_doc.py [options]
```

### Command Line Arguments

- `-o`, `--output` : The output directory for generated files (defaults to `output`).
- `-c`, `--count` : Number of document pairs to generate (defaults to `1`).
- `--ollama` : Use Ollama to generate context-appropriate text instead of random words (requires the `ollama` package).
- `--ollama-model` : Specify an Ollama model to use, for example `llama3` (defaults to `llama3`).

### Examples

**Generate 5 random documents into the `output` directory:**
```bash
python generate_doc.py -c 5
```

**Generate 10 documents into a specific `data/` directory using Ollama (Llama 3):**
```bash
python generate_doc.py -c 10 -o data --ollama --ollama-model llama3
```

*(If you use Ollama, ensure the Ollama server is running locally and the model `llama3` is pulled: `ollama run llama3`)*
