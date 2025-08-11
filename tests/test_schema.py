# =============================
# tests/test_schema.py (schema lock tests)
# =============================
from pathlib import Path
import subprocess
import pandas as pd


def _write_min_boundaries(dirpath: Path):
    # Minimal square for both nuclei and cells (id=1)
    nuc = pd.DataFrame(
        {
            "nucleus_id": [1, 1, 1, 1],
            "vertex_x": [0.0, 1.0, 1.0, 0.0],
            "vertex_y": [0.0, 0.0, 1.0, 1.0],
        }
    )
    cell = pd.DataFrame(
        {
            "cell_id": [1, 1, 1, 1],
            "vertex_x": [0.0, 1.0, 1.0, 0.0],
            "vertex_y": [0.0, 0.0, 1.0, 1.0],
        }
    )
    nuc.to_parquet(dirpath / "nucleus_boundaries.parquet")
    cell.to_parquet(dirpath / "cell_boundaries.parquet")


def test_schema_nuclei(tmp_path: Path):
    _write_min_boundaries(tmp_path)
    out = tmp_path / "nuc.parquet"
    r = subprocess.run(
        [
            "xenmorph",
            "metrics",
            "compute",
            str(tmp_path),
            "--scope",
            "nuclei",
            "--out",
            str(out),
        ],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stderr
    df = pd.read_parquet(out)
    expect = {
        "nucleus_id",
        "scope",
        "nuclear_area_um2",
        "nuclear_perimeter_um",
        "nuclear_equivalent_diameter_um",
        "nuclear_circularity",
        "nuclear_aspect_ratio",
        "nuclear_elongation",
        "nuclear_rectangularity",
        "nuclear_solidity",
        "nuclear_convexity",
        "nuclear_has_holes",
        "nuclear_is_valid",
        "nuclear_num_holes",
        "nuclear_touches_border",
    }
    assert expect.issubset(df.columns)


def test_schema_cells(tmp_path: Path):
    _write_min_boundaries(tmp_path)
    out = tmp_path / "cell.parquet"
    r = subprocess.run(
        [
            "xenmorph",
            "metrics",
            "compute",
            str(tmp_path),
            "--scope",
            "cells",
            "--out",
            str(out),
        ],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stderr
    df = pd.read_parquet(out)
    expect = {
        "cell_id",
        "scope",
        "cell_area_um2",
        "cell_perimeter_um",
        "cell_equivalent_diameter_um",
        "cell_circularity",
        "cell_aspect_ratio",
        "cell_elongation",
        "cell_rectangularity",
        "cell_solidity",
        "cell_convexity",
        "cell_has_holes",
        "cell_is_valid",
        "cell_num_holes",
        "cell_touches_border",
    }
    assert expect.issubset(df.columns)


def test_schema_both_wide(tmp_path: Path):
    _write_min_boundaries(tmp_path)
    out = tmp_path / "both.parquet"
    r = subprocess.run(
        ["xenmorph", "metrics", "compute", str(tmp_path), "--out", str(out)],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stderr
    df = pd.read_parquet(out)
    # At minimum we expect a per-cell row with both cell_ and nuclear_ metric columns
    expect_min = {"cell_id", "cell_circularity", "nuclear_circularity"}
    assert expect_min.issubset(df.columns)
