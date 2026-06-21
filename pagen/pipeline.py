"""Fused document generation pipeline.

Render → augment → write one final image per page.  No intermediate clean
copy is ever written.

Two generation modes:
  - dataset: produces train/val splits with polygon labels.json (doctr format)
  - eval: produces image + plain-text ground truth per doc, no polygons
"""

from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
import random
import re
from dataclasses import dataclass, field
from typing import Optional

import cv2


# ---------------------------------------------------------------------------
# Worker task / result types
# ---------------------------------------------------------------------------

@dataclass
class DatasetTask:
    output_dir: str
    file_id: int
    template_paths: list[str]
    fonts: list[str]
    words: list[str]
    dpi: int
    augment: bool
    augment_ctx: Optional[object]   # AugmentContext or None
    llm_config: Optional[object]    # LLMConfig or None
    keep_txt: bool
    keep_pdf: bool
    seed: int


@dataclass
class EvalTask:
    output_dir: str
    file_id: int
    template_paths: list[str]
    fonts: list[str]
    words: list[str]
    dpi: int
    llm_config: Optional[object]
    seed: int


# ---------------------------------------------------------------------------
# Dataset worker
# ---------------------------------------------------------------------------

def _dataset_worker(task: DatasetTask):
    random.seed(task.seed)
    rng = random.Random(task.seed)

    from pagen.render import render_document
    from pagen.augment import augment_page

    template_path = rng.choice(task.template_paths)
    with open(template_path, encoding="utf-8") as f:
        template_content = f.read()

    try:
        pages = render_document(
            template_content,
            fonts=task.fonts,
            words=task.words,
            dpi=task.dpi,
            llm_config=task.llm_config,
            keep_pdf=task.keep_pdf,
        )
    except Exception as e:
        print(f"  WARNING: render failed for doc {task.file_id}: {e}")
        return []

    if not pages:
        return []

    images_dir = os.path.join(task.output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    n_pages = len(pages)
    results = []

    for page_idx, page in enumerate(pages):
        page_suffix = f"_p{page_idx + 1}" if n_pages > 1 else ""
        page_base = f"{task.file_id}{page_suffix}"

        if task.augment and task.augment_ctx is not None:
            # Decode PNG to BGR ndarray, augment, re-encode
            img_array = cv2.imdecode(
                __import__("numpy").frombuffer(page.png_bytes, __import__("numpy").uint8),
                cv2.IMREAD_COLOR,
            )
            img_array, new_polygons = augment_page(
                img_array, page.polygons, task.augment_ctx, rng
            )
            success, enc = cv2.imencode(".png", img_array)
            if not success:
                print(f"  WARNING: PNG encode failed for doc {task.file_id} page {page_idx}")
                continue
            final_png = enc.tobytes()
            final_polygons = new_polygons
            h, w = img_array.shape[:2]
        else:
            final_png = page.png_bytes
            final_polygons = page.polygons
            w, h = page.width, page.height

        # Write image
        png_path = os.path.join(images_dir, f"{page_base}.png")
        with open(png_path, "wb") as f:
            f.write(final_png)

        # Optional: txt and pdf (only on first page for multi-page docs)
        if task.keep_txt:
            txt_path = os.path.join(task.output_dir, f"{page_base}.txt")
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write("\n".join(page.labels))

        if task.keep_pdf and page.pdf_bytes and page_idx == 0:
            pdf_path = os.path.join(task.output_dir, f"{task.file_id}.pdf")
            with open(pdf_path, "wb") as f:
                f.write(page.pdf_bytes)

        entry = {
            "img_dimensions": [w, h],
            "img_hash": hashlib.sha256(final_png).hexdigest(),
            "polygons": final_polygons,
            "labels": page.labels,
        }
        results.append((f"{page_base}.png", entry))

    return results


def _dataset_worker_unpack(task):
    random.seed()   # reseed per-worker after fork
    try:
        return _dataset_worker(task)
    except Exception as e:
        print(f"  WARNING: worker exception: {e}")
        return []


# ---------------------------------------------------------------------------
# Eval worker
# ---------------------------------------------------------------------------

def _eval_worker(task: EvalTask):
    random.seed(task.seed)
    rng = random.Random(task.seed)

    from pagen.render import render_document

    template_path = rng.choice(task.template_paths)
    with open(template_path, encoding="utf-8") as f:
        template_content = f.read()

    try:
        pages = render_document(
            template_content,
            fonts=task.fonts,
            words=task.words,
            dpi=task.dpi,
            llm_config=task.llm_config,
        )
    except Exception as e:
        print(f"  WARNING: render failed for doc {task.file_id}: {e}")
        return []

    os.makedirs(task.output_dir, exist_ok=True)
    results = []

    for page_idx, page in enumerate(pages):
        n_pages = len(pages)
        page_suffix = f"_p{page_idx + 1}" if n_pages > 1 else ""
        page_base = f"{task.file_id}{page_suffix}"

        png_path = os.path.join(task.output_dir, f"{page_base}.png")
        with open(png_path, "wb") as f:
            f.write(page.png_bytes)

        txt_path = os.path.join(task.output_dir, f"{page_base}.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(page.plain_text)

        results.append(page_base)

    return results


def _eval_worker_unpack(task):
    random.seed()
    try:
        return _eval_worker(task)
    except Exception as e:
        print(f"  WARNING: eval worker exception: {e}")
        return []


# ---------------------------------------------------------------------------
# ID assignment helpers (append-safe)
# ---------------------------------------------------------------------------

def _next_id(images_dir: str) -> int:
    """Return the next available file ID by scanning existing PNGs."""
    if not os.path.isdir(images_dir):
        return 1
    existing = [
        int(m.group(1))
        for f in os.listdir(images_dir)
        if (m := re.match(r"^(\d+)(?:_p\d+)?\.png$", f))
    ]
    return max(existing, default=0) + 1


def _next_id_plain(output_dir: str) -> int:
    """Return next ID for eval output (no images/ subfolder)."""
    existing = [
        int(m.group(1))
        for f in os.listdir(output_dir)
        if (m := re.match(r"^(\d+)(?:_p\d+)?\.png$", f))
    ] if os.path.isdir(output_dir) else []
    return max(existing, default=0) + 1


# ---------------------------------------------------------------------------
# Incremental JSON writer
# ---------------------------------------------------------------------------

class _JsonWriter:
    def __init__(self, path: str, existing: dict):
        self._f = open(path, "w", encoding="utf-8")
        self._first = True
        self._f.write("{\n")
        for k, v in existing.items():
            self._emit(k, v)

    def _emit(self, key, value):
        sep = "" if self._first else ",\n"
        self._f.write(sep + json.dumps(key) + ": " + json.dumps(value, ensure_ascii=False))
        self._first = False

    def write(self, key, value):
        self._emit(key, value)

    def close(self):
        self._f.write("\n}\n")
        self._f.close()


# ---------------------------------------------------------------------------
# Public: generate_split
# ---------------------------------------------------------------------------

def generate_split(
    output_dir: str,
    count: int,
    template_paths: list[str],
    fonts: list[str],
    words: list[str],
    dpi: int = 150,
    augment: bool = True,
    augment_ctx=None,
    llm_config=None,
    keep_txt: bool = False,
    keep_pdf: bool = False,
    workers: int = 1,
    seed: int = 42,
) -> None:
    """Generate ``count`` documents into ``output_dir`` (doctr detection format)."""
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    start_id = _next_id(images_dir)
    file_ids = list(range(start_id, start_id + count))

    tasks = [
        DatasetTask(
            output_dir=output_dir,
            file_id=fid,
            template_paths=template_paths,
            fonts=fonts,
            words=words,
            dpi=dpi,
            augment=augment,
            augment_ctx=augment_ctx,
            llm_config=llm_config,
            keep_txt=keep_txt,
            keep_pdf=keep_pdf,
            seed=(seed * 1_000_003 + i) % (2**31),
        )
        for i, fid in enumerate(file_ids)
    ]

    labels_path = os.path.join(output_dir, "labels.json")
    existing = {}
    if os.path.exists(labels_path):
        with open(labels_path, encoding="utf-8") as f:
            existing = json.load(f)

    n_workers = max(1, min(workers, count))
    done = 0

    writer = _JsonWriter(labels_path, existing)
    pool = multiprocessing.Pool(n_workers)
    try:
        for page_results in pool.imap_unordered(_dataset_worker_unpack, tasks):
            for img_name, entry in page_results:
                writer.write(img_name, entry)
            if page_results:
                done += 1
            print(f"  {done}/{count} done", end="\r", flush=True)
        pool.close()
        pool.join()
    except KeyboardInterrupt:
        print("\nInterrupted — terminating workers…")
        pool.terminate()
        pool.join()
    finally:
        writer.close()

    print(f"\n{output_dir}: {done} documents, labels.json updated")


# ---------------------------------------------------------------------------
# Public: generate_eval
# ---------------------------------------------------------------------------

def generate_eval(
    output_dir: str,
    count: int,
    template_paths: list[str],
    fonts: list[str],
    words: list[str],
    dpi: int = 150,
    llm_config=None,
    workers: int = 1,
    seed: int = 42,
) -> None:
    """Generate ``count`` eval documents (PNG + plain-text GT, no polygons)."""
    os.makedirs(output_dir, exist_ok=True)
    start_id = _next_id_plain(output_dir)

    tasks = [
        EvalTask(
            output_dir=output_dir,
            file_id=start_id + i,
            template_paths=template_paths,
            fonts=fonts,
            words=words,
            dpi=dpi,
            llm_config=llm_config,
            seed=(seed * 1_000_003 + i) % (2**31),
        )
        for i in range(count)
    ]

    n_workers = max(1, min(workers, count))
    done = 0

    pool = multiprocessing.Pool(n_workers)
    try:
        for result in pool.imap_unordered(_eval_worker_unpack, tasks):
            done += len(result)
            print(f"  {done}/{count} pages done", end="\r", flush=True)
        pool.close()
        pool.join()
    except KeyboardInterrupt:
        print("\nInterrupted — terminating workers…")
        pool.terminate()
        pool.join()
    finally:
        pass

    print(f"\n{output_dir}: {done} page(s) generated")
