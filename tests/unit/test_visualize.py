"""Unit tests for pagen.visualize.validate_entry (script-agnostic)."""

from pagen import visualize


def _entry(polygons, labels, dims):
    return {"polygons": polygons, "labels": labels, "img_dimensions": dims}


def test_validate_clean_entry():
    info = _entry([[[0, 0], [1, 0], [1, 1], [0, 1]]], ["x"], [10, 10])
    assert visualize.validate_entry("a.png", info, (10, 10)) == []


def test_validate_dimension_mismatch():
    info = _entry([], ["x"], [10, 10])
    problems = visualize.validate_entry("a.png", info, (20, 20))
    assert any("img_dimensions" in p for p in problems)


def test_validate_out_of_bounds_point():
    info = _entry([[[0, 0], [99, 0], [99, 99], [0, 99]]], ["x"], [10, 10])
    problems = visualize.validate_entry("a.png", info, (10, 10))
    assert any("out of bounds" in p for p in problems)


def test_validate_polygon_label_count_mismatch():
    info = _entry([[[0, 0], [1, 0], [1, 1], [0, 1]]], ["x", "y"], [10, 10])
    problems = visualize.validate_entry("a.png", info, (10, 10))
    assert any("polygons but" in p for p in problems)


def test_validate_no_labels():
    info = _entry([], [], [10, 10])
    problems = visualize.validate_entry("a.png", info, (10, 10))
    assert "no labels" in problems


def test_validate_empty_label():
    info = _entry([[[0, 0], [1, 0], [1, 1], [0, 1]]], ["  "], [10, 10])
    problems = visualize.validate_entry("a.png", info, (10, 10))
    assert any("empty label" in p for p in problems)
