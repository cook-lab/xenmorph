# =============================
# tests/test_cli.py (end-to-end CLI, clean shape)
# =============================
from pathlib import Path
import subprocess
import pandas as pd


def _write_min_nucleus_boundaries(dirpath: Path):
    df = pd.DataFrame(
        {
            "nucleus_id": [1, 1, 1, 1],
            "vertex_x": [0.0, 1.0, 1.0, 0.0],
            "vertex_y": [0.0, 0.0, 1.0, 1.0],
        }
    )
    df.to_parquet(dirpath / "nucleus_boundaries.parquet")


def test_single_command_runs_and_writes(tmp_path: Path):
    _write_min_nucleus_boundaries(tmp_path)
    out = tmp_path / "out.parquet"
    result = subprocess.run(
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
    assert result.returncode == 0, result.stderr

    df_out = pd.read_parquet(out)
    assert not df_out.empty
    # New schema uses prefixed columns and explicit ID names per scope
    assert {"nucleus_id", "nuclear_circularity"}.issubset(df_out.columns)
