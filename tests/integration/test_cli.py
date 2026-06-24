"""CLI tests: argument parsing, llm-config building, resource loading,
dispatch routing, subcommand guards, and a render-free e2e (visualize).

Heavy commands are not actually run — _cmd_* handlers are mocked where the
point is routing, and the one e2e path (visualize) needs no rendering.
"""

import argparse
import json
import sys

import pytest

from pagen import cli


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def test_parser_dataset_defaults():
    args = cli._build_parser().parse_args(["dataset", "--train", "5"])
    assert args.subcommand == "dataset"
    assert args.train == 5
    assert args.dpi == 150
    assert args.val is None


def test_parser_eval():
    args = cli._build_parser().parse_args(["eval", "-c", "3", "-o", "out/e"])
    assert args.count == 3 and args.output == "out/e"


def test_parser_visualize():
    args = cli._build_parser().parse_args(["visualize", "somepath", "--max", "4"])
    assert args.path == "somepath" and args.max == 4


def test_parser_corners_edit():
    args = cli._build_parser().parse_args(["corners", "--edit"])
    assert args.edit is True and args.visualize is False


def test_parser_templates():
    args = cli._build_parser().parse_args(["templates", "عقد", "--random", "2"])
    assert args.doc_types == ["عقد"] and args.random_n == 2


# ---------------------------------------------------------------------------
# _llm_config_from_args
# ---------------------------------------------------------------------------

def test_llm_config_none_without_flag():
    assert cli._llm_config_from_args(argparse.Namespace(llm=False)) is None


def test_llm_config_built_with_flag():
    ns = argparse.Namespace(llm=True, llm_base_url="http://x/v1",
                            llm_model="m", api_key_env="K")
    cfg = cli._llm_config_from_args(ns)
    assert (cfg.base_url, cfg.model, cfg.api_key_env) == ("http://x/v1", "m", "K")


# ---------------------------------------------------------------------------
# _load_resources
# ---------------------------------------------------------------------------

def test_load_resources_missing_templates_dir_exits(tmp_path):
    ns = argparse.Namespace(templates_dir=str(tmp_path / "nope"),
                            fonts_dir=str(tmp_path), corpus=None)
    with pytest.raises(SystemExit):
        cli._load_resources(ns)


def test_load_resources_no_md_exits(tmp_path):
    ns = argparse.Namespace(templates_dir=str(tmp_path),
                            fonts_dir=str(tmp_path), corpus=None)
    with pytest.raises(SystemExit):
        cli._load_resources(ns)


def test_load_resources_ok(tmp_path):
    (tmp_path / "a.md").write_text("# {WORDS_1}", encoding="utf-8")
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("alpha beta", encoding="utf-8")
    ns = argparse.Namespace(templates_dir=str(tmp_path),
                            fonts_dir=str(tmp_path / "nofonts"),
                            corpus=str(corpus))
    templates, fonts, words = cli._load_resources(ns)
    assert len(templates) == 1
    assert fonts == []                  # missing fonts dir -> empty
    assert words == ["alpha", "beta"]


# ---------------------------------------------------------------------------
# main() dispatch
# ---------------------------------------------------------------------------

def test_main_dispatch_eval(monkeypatch):
    called = {}
    monkeypatch.setattr(cli, "_cmd_eval", lambda args: called.setdefault("eval", args))
    monkeypatch.setattr(sys, "argv", ["pagen", "eval", "-c", "2"])
    cli.main()
    assert called["eval"].count == 2


def test_main_dispatch_visualize(monkeypatch):
    called = {}
    monkeypatch.setattr(cli, "_cmd_visualize", lambda args: called.setdefault("v", args))
    monkeypatch.setattr(sys, "argv", ["pagen", "visualize", "p"])
    cli.main()
    assert "v" in called


def test_main_explicit_dataset_defaults_missing_count_to_zero(monkeypatch):
    called = {}
    monkeypatch.setattr(cli, "_cmd_dataset", lambda args: called.setdefault("d", args))
    monkeypatch.setattr(sys, "argv", ["pagen", "dataset", "--train", "1"])
    cli.main()
    assert called["d"].train == 1 and called["d"].val == 0


def test_main_flags_without_subcommand_defaults_to_dataset(monkeypatch):
    # Regression test for the CLI default-subcommand fix: `pagen --train 1`
    # (flags, no explicit 'dataset') must route to the dataset command instead
    # of erroring with "invalid choice: '1'".
    called = {}
    monkeypatch.setattr(cli, "_cmd_dataset", lambda args: called.setdefault("d", args))
    monkeypatch.setattr(sys, "argv", ["pagen", "--train", "1"])
    cli.main()
    assert called["d"].train == 1 and called["d"].val == 0


def test_main_bare_non_tty_still_errors(monkeypatch):
    # Bare `pagen` with no flags on a non-TTY should still ask for counts
    # (it defaults to the dataset subcommand, then hits the wizard guard).
    monkeypatch.setattr(sys, "argv", ["pagen"])
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    with pytest.raises(SystemExit):
        cli.main()


def test_main_dataset_zero_counts_errors(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["pagen", "dataset", "--train", "0", "--val", "0"])
    with pytest.raises(SystemExit):
        cli.main()


def test_main_wizard_non_tty_errors(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["pagen", "dataset"])
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    with pytest.raises(SystemExit):
        cli.main()


# ---------------------------------------------------------------------------
# _cmd_templates guards
# ---------------------------------------------------------------------------

def test_cmd_templates_requires_llm():
    with pytest.raises(SystemExit):
        cli._cmd_templates(argparse.Namespace(llm=False))


def test_cmd_templates_slug_with_multiple_exits(tmp_path):
    ns = argparse.Namespace(
        llm=True, llm_base_url="http://x/v1", llm_model="m", api_key_env="K",
        random_n=None, doc_types=["a", "b"], file=None, slug="s", output=str(tmp_path),
    )
    with pytest.raises(SystemExit):
        cli._cmd_templates(ns)


def test_cmd_templates_empty_work_exits(tmp_path):
    ns = argparse.Namespace(
        llm=True, llm_base_url="http://x/v1", llm_model="m", api_key_env="K",
        random_n=None, doc_types=[], file=None, slug=None, output=str(tmp_path),
    )
    with pytest.raises(SystemExit):
        cli._cmd_templates(ns)


# ---------------------------------------------------------------------------
# End-to-end: visualize a tiny dataset (no rendering required)
# ---------------------------------------------------------------------------

def test_cli_visualize_end_to_end(tmp_path, monkeypatch):
    from PIL import Image

    ds = tmp_path / "ds"
    (ds / "images").mkdir(parents=True)
    Image.new("RGB", (20, 20), "white").save(ds / "images" / "1.png")
    labels = {
        "1.png": {
            "img_dimensions": [20, 20],
            "img_hash": "x",
            "polygons": [[[0, 0], [5, 0], [5, 5], [0, 5]]],
            "labels": ["مرحبا"],
        }
    }
    (ds / "labels.json").write_text(json.dumps(labels, ensure_ascii=False), encoding="utf-8")

    save = tmp_path / "out"
    monkeypatch.setattr(sys, "argv",
                        ["pagen", "visualize", str(ds), "--save", str(save)])
    cli.main()
    assert (save / "1.png").exists()
