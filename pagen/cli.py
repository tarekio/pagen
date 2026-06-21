"""pagen command-line interface.

Subcommands:
  pagen [dataset]   Generate a train+val detection dataset (default)
  pagen eval        Generate eval images + plain-text ground truth
  pagen templates   Generate markdown document templates via LLM
  pagen corners     Manage paper corner cache (detect / edit / visualize)
  pagen visualize   Validate and overlay a produced detection dataset

Run `pagen <subcommand> --help` for per-command options.
"""

from __future__ import annotations

import argparse
import os
import sys

from pagen._paths import (
    DEFAULT_FONT,
    FONTS_DIR,
    SCENE_DIR,
    TEMPLATES_DIR,
    TEXTURES_DIR,
)


# ---------------------------------------------------------------------------
# Shared LLM argument group
# ---------------------------------------------------------------------------

def _add_llm_args(parser: argparse.ArgumentParser):
    g = parser.add_argument_group("LLM options")
    g.add_argument("--llm", action="store_true",
                   help="Use LLM to fill template content instead of random corpus words")
    g.add_argument("--llm-base-url", default="http://localhost:11434/v1",
                   metavar="URL",
                   help="OpenAI-compatible API base URL (default: Ollama at localhost:11434)")
    g.add_argument("--llm-model", default="llama3", metavar="MODEL",
                   help="Model name (default: llama3)")
    g.add_argument("--api-key-env", default="OPENAI_API_KEY", metavar="VAR",
                   help="Env var holding the API key; ignored for Ollama (default: OPENAI_API_KEY)")


def _llm_config_from_args(args):
    if not getattr(args, "llm", False):
        return None
    from pagen.llm import LLMConfig, _try_load_dotenv
    _try_load_dotenv()
    return LLMConfig(
        base_url=args.llm_base_url,
        model=args.llm_model,
        api_key_env=args.api_key_env,
    )


# ---------------------------------------------------------------------------
# Shared render / resource setup
# ---------------------------------------------------------------------------

def _load_resources(args):
    """Return (template_paths, fonts, words) from args."""
    from pagen.corpus import load_words
    from pagen.fonts import list_fonts

    templates_dir = getattr(args, "templates_dir", TEMPLATES_DIR)
    if not os.path.isdir(templates_dir):
        sys.exit(f"ERROR: templates directory not found: {templates_dir}")
    template_paths = [
        os.path.join(templates_dir, f)
        for f in os.listdir(templates_dir)
        if f.endswith(".md")
    ]
    if not template_paths:
        sys.exit(f"ERROR: no .md templates found in {templates_dir}")

    fonts_dir = getattr(args, "fonts_dir", FONTS_DIR)
    fonts = list_fonts(fonts_dir)

    corpus = getattr(args, "corpus", None)
    words = load_words(corpus)

    return template_paths, fonts, words


# ---------------------------------------------------------------------------
# Interactive wizard
# ---------------------------------------------------------------------------

def _wizard() -> argparse.Namespace:
    """Prompt the user for dataset generation options when none are given."""
    print("=== pagen dataset wizard ===")
    print("Press Enter to accept defaults.\n")

    def _ask(prompt, default):
        val = input(f"  {prompt} [{default}]: ").strip()
        return val if val else str(default)

    train = int(_ask("Train documents", 100))
    val = int(_ask("Val documents", 20))
    output = _ask("Output directory", "output")
    pdf_only = _ask("PDF-only (no augmentation)? [y/N]", "N").lower() in ("y", "yes")
    keep_txt = _ask("Save plain-text ground truth (.txt)? [y/N]", "N").lower() in ("y", "yes")
    keep_pdf = _ask("Save PDF files? [y/N]", "N").lower() in ("y", "yes")
    use_llm = _ask("Use LLM for content fill? [y/N]", "N").lower() in ("y", "yes")

    ns = argparse.Namespace(
        train=train, val=val, output=output,
        pdf_only=pdf_only, keep_txt=keep_txt, keep_pdf=keep_pdf,
        llm=use_llm,
        llm_base_url="http://localhost:11434/v1",
        llm_model="llama3",
        api_key_env="OPENAI_API_KEY",
        dpi=150,
        workers=os.cpu_count() or 4,
        seed=42,
        clean_prob=0.10, textures_prob=0.45, scene_prob=0.45,
        scene_dir=SCENE_DIR,
        textures_dir=TEXTURES_DIR,
        templates_dir=TEMPLATES_DIR,
        corpus=None,
        fonts_dir=FONTS_DIR,
    )
    return ns


# ---------------------------------------------------------------------------
# Subcommand: dataset
# ---------------------------------------------------------------------------

def _cmd_dataset(args):
    from pagen.augment import AugmentContext, ensure_corners_cache
    from pagen.pipeline import generate_split

    template_paths, fonts, words = _load_resources(args)
    llm_config = _llm_config_from_args(args)

    augment = not args.pdf_only
    augment_ctx = None

    if augment:
        total = args.clean_prob + args.textures_prob + args.scene_prob
        clean_prob = args.clean_prob / total
        scan_prob = args.textures_prob / total

        scan_exts = {".jpg", ".jpeg", ".png", ".bmp"}
        scan_imgs = (
            [os.path.join(args.textures_dir, f)
             for f in os.listdir(args.textures_dir)
             if os.path.splitext(f)[1].lower() in scan_exts]
            if os.path.isdir(args.textures_dir) else []
        )

        corners_cache, pic_imgs = {}, []
        if os.path.isdir(args.scene_dir):
            print("Reconciling paper corner cache…")
            # Appends auto-detected corners for any NEW images only; existing
            # entries (and user-edited ones) are never overwritten. Steady-state
            # runs with no new images leave the cache file untouched.
            corners_cache = ensure_corners_cache(args.scene_dir)
            pic_imgs = [os.path.join(args.scene_dir, f) for f in corners_cache]

        print(f"Scan textures: {len(scan_imgs)}, photo backgrounds: {len(pic_imgs)}")
        augment_ctx = AugmentContext(
            scan_imgs=scan_imgs,
            corners_cache=corners_cache,
            pic_imgs=pic_imgs,
            clean_prob=clean_prob,
            scan_prob=scan_prob,
        )

    common = dict(
        template_paths=template_paths,
        fonts=fonts,
        words=words,
        dpi=args.dpi,
        augment=augment,
        augment_ctx=augment_ctx,
        llm_config=llm_config,
        keep_txt=args.keep_txt,
        keep_pdf=args.keep_pdf,
        workers=args.workers,
        seed=args.seed,
    )

    if args.train > 0:
        train_dir = os.path.join(args.output, "train")
        print(f"\nGenerating {args.train} training documents → {train_dir}")
        generate_split(train_dir, args.train, **common)

    if args.val > 0:
        val_dir = os.path.join(args.output, "val")
        print(f"\nGenerating {args.val} validation documents → {val_dir}")
        generate_split(val_dir, args.val, **common)

    print("\nDone.")


# ---------------------------------------------------------------------------
# Subcommand: eval
# ---------------------------------------------------------------------------

def _cmd_eval(args):
    from pagen.pipeline import generate_eval

    template_paths, fonts, words = _load_resources(args)
    llm_config = _llm_config_from_args(args)

    print(f"Generating {args.count} eval documents → {args.output}")
    generate_eval(
        output_dir=args.output,
        count=args.count,
        template_paths=template_paths,
        fonts=fonts,
        words=words,
        dpi=args.dpi,
        llm_config=llm_config,
        workers=args.workers,
        seed=args.seed,
    )


# ---------------------------------------------------------------------------
# Subcommand: templates
# ---------------------------------------------------------------------------

def _cmd_templates(args):
    from pagen.llm import LLMConfig, _try_load_dotenv
    from pagen.templates import pick_random, run_generation, _slugify

    if not args.llm:
        sys.exit("ERROR: template generation requires --llm (an LLM must be used to create templates)")

    _try_load_dotenv()
    llm_config = LLMConfig(
        base_url=args.llm_base_url,
        model=args.llm_model,
        api_key_env=args.api_key_env,
    )

    work: list[tuple[str, str]] = []

    if args.random_n:
        work += pick_random(args.random_n, args.output)

    for name in args.doc_types:
        work.append((name, args.slug if args.slug else _slugify(name)))

    if args.file:
        with open(args.file, encoding="utf-8") as f:
            for line in f:
                name = line.strip()
                if name:
                    work.append((name, _slugify(name)))

    if not work:
        sys.exit("ERROR: provide doc type(s), --random N, or --file FILE")

    if args.slug and len(work) > 1:
        sys.exit("ERROR: --slug can only be used with a single doc type")

    run_generation(work, llm_config, args.output)


# ---------------------------------------------------------------------------
# Subcommand: corners
# ---------------------------------------------------------------------------

def _cmd_corners(args):
    from pagen.corners import build_cache, launch_editor, export_overlays

    if args.edit:
        launch_editor(args.scene_dir, args.max_dim)
    elif args.visualize:
        out_dir = args.out or os.path.join(args.scene_dir, "corners_vis")
        export_overlays(args.scene_dir, out_dir)
    else:
        cache = build_cache(args.scene_dir)
        print(f"Cache has {len(cache)} entries in {args.scene_dir}")


# ---------------------------------------------------------------------------
# Subcommand: visualize
# ---------------------------------------------------------------------------

def _cmd_visualize(args):
    from pagen.visualize import visualize_dataset

    visualize_dataset(
        path=args.path,
        show_labels=not args.no_labels,
        max_images=args.max,
        debug=args.debug,
        show=args.show,
        save_dir=args.save,
        seed=args.seed,
        font_path=args.font,
    )


# ---------------------------------------------------------------------------
# Argument parser construction
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pagen",
        description="Arabic document image dataset generator.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="subcommand")

    # -- dataset (default) --------------------------------------------------
    ds = sub.add_parser("dataset", help="Generate train+val detection dataset (default)")
    ds.add_argument("--train", type=int, default=None, metavar="N",
                    help="Number of training documents")
    ds.add_argument("--val", type=int, default=None, metavar="N",
                    help="Number of validation documents")
    ds.add_argument("-o", "--output", default="output", metavar="DIR")
    ds.add_argument("--pdf-only", action="store_true",
                    help="Skip augmentation; produce clean rendered images")
    ds.add_argument("--keep-txt", action="store_true",
                    help="Save plain-text word labels per page (.txt)")
    ds.add_argument("--keep-pdf", action="store_true",
                    help="Save the intermediate PDF per document")
    ds.add_argument("--dpi", type=int, default=150)
    ds.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    ds.add_argument("--seed", type=int, default=42)
    ds.add_argument("--templates-dir", default=TEMPLATES_DIR, metavar="DIR")
    ds.add_argument("--corpus", default=None, metavar="PATH",
                    help="Corpus file or directory (default: resources/corpora/)")
    ds.add_argument("--fonts-dir", default=FONTS_DIR, metavar="DIR")
    # Augment knobs
    ag = ds.add_argument_group("Augmentation")
    ag.add_argument("--clean-prob", type=float, default=0.10)
    ag.add_argument("--textures-prob", type=float, default=0.45)
    ag.add_argument("--scene-prob", type=float, default=0.45)
    ag.add_argument("--scene-dir", default=SCENE_DIR, metavar="DIR")
    ag.add_argument("--textures-dir", default=TEXTURES_DIR, metavar="DIR")
    _add_llm_args(ds)

    # -- eval ---------------------------------------------------------------
    ev = sub.add_parser("eval", help="Generate eval images + plain-text ground truth")
    ev.add_argument("-c", "--count", type=int, default=10, metavar="N")
    ev.add_argument("-o", "--output", default="output/eval", metavar="DIR")
    ev.add_argument("--dpi", type=int, default=150)
    ev.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    ev.add_argument("--seed", type=int, default=42)
    ev.add_argument("--templates-dir", default=TEMPLATES_DIR, metavar="DIR")
    ev.add_argument("--corpus", default=None, metavar="PATH")
    ev.add_argument("--fonts-dir", default=FONTS_DIR, metavar="DIR")
    _add_llm_args(ev)

    # -- templates ----------------------------------------------------------
    tp = sub.add_parser("templates", help="Generate markdown document templates via LLM")
    tp.add_argument("doc_types", nargs="*", metavar="DOC_TYPE")
    tp.add_argument("--random", "-r", type=int, dest="random_n", metavar="N",
                    help="Generate N templates randomly from the built-in pool")
    tp.add_argument("--file", "-f", metavar="FILE",
                    help="Text file with one document type per line")
    tp.add_argument("--slug", help="Filename slug (single doc type only)")
    tp.add_argument("-o", "--output", default=TEMPLATES_DIR, metavar="DIR")
    _add_llm_args(tp)

    # -- corners ------------------------------------------------------------
    co = sub.add_parser("corners", help="Manage paper corner cache for photo backgrounds")
    co.add_argument("--scene-dir", default=SCENE_DIR, metavar="DIR")
    co.add_argument("--edit", action="store_true", help="Launch interactive editor")
    co.add_argument("--visualize", action="store_true", help="Export corner overlay images")
    co.add_argument("--out", default=None, metavar="DIR",
                    help="Output dir for overlays (default: <scene-dir>/corners_vis)")
    co.add_argument("--max-dim", type=int, default=1100,
                    help="Max display dimension for editor")

    # -- visualize ----------------------------------------------------------
    vi = sub.add_parser("visualize", help="Validate and overlay a detection dataset")
    vi.add_argument("path", help="Dataset directory (contains labels.json and images/)")
    vi.add_argument("--max", type=int, default=10, metavar="N")
    vi.add_argument("--save", default=None, metavar="DIR")
    vi.add_argument("--show", action="store_true")
    vi.add_argument("--no-labels", action="store_true")
    vi.add_argument("--seed", type=int, default=None)
    vi.add_argument("--debug", action="store_true")
    vi.add_argument("--font", default=None, metavar="PATH")

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = _build_parser()
    args = parser.parse_args()

    # Default subcommand = dataset; may trigger wizard
    if args.subcommand is None or args.subcommand == "dataset":
        if args.subcommand is None:
            # Re-parse with 'dataset' defaults
            args = _build_parser().parse_args(["dataset"] + sys.argv[1:])

        # Interactive wizard when train/val are unset and we're on a TTY
        if args.train is None and args.val is None:
            if sys.stdin.isatty():
                args = _wizard()
            else:
                parser.error("Provide --train N and/or --val N, or run interactively.")

        # Apply defaults for whichever count was omitted
        if args.train is None:
            args.train = 0
        if args.val is None:
            args.val = 0

        if args.train == 0 and args.val == 0:
            parser.error("--train and/or --val must be > 0")

        _cmd_dataset(args)

    elif args.subcommand == "eval":
        _cmd_eval(args)

    elif args.subcommand == "templates":
        _cmd_templates(args)

    elif args.subcommand == "corners":
        _cmd_corners(args)

    elif args.subcommand == "visualize":
        font_path = args.font
        if font_path is None:
            # Try to find Amiri-Regular in the fonts dir
            font_path = DEFAULT_FONT if os.path.exists(DEFAULT_FONT) else None
        args.font = font_path
        _cmd_visualize(args)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
