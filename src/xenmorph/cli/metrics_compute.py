from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import pandas as pd
import typer

from xenmorph.io.xenium import (
    frame_from_extents,
    load_frame_um,
    read_boundaries,
    read_cells_table,
)
from xenmorph.metrics.core import compute_metrics

app = typer.Typer(help="Compute morphometrics from a Xenium output bundle in one pass.")


def _prefix_columns(df: pd.DataFrame, prefix: str, id_name: str) -> pd.DataFrame:
    out = df.copy()
    out = out.rename(columns={"object_id": id_name})
    metric_cols = [c for c in out.columns if c not in {id_name, "scope"}]
    out.rename(columns={c: f"{prefix}{c}" for c in metric_cols}, inplace=True)
    return out


@app.command("compute")
def compute(
    xenium_dir: Path = typer.Argument(
        ...,
        exists=True,
        readable=True,
        help="Path to Xenium output directory (region outs)",
    ),
    scope: str = typer.Option(
        "both",
        "--scope",
        case_sensitive=False,
        help="nuclei|cells|both (default: both)",
    ),
    out_path: Optional[Path] = typer.Option(
        None,
        "--out",
        help="Output file (.parquet or .csv). Default: <outs>/morphometrics/metrics.parquet",
    ),
    metrics: List[str] = typer.Option(
        None, help="Subset of metrics to compute; default=all"
    ),
    border_eps_um: float = typer.Option(
        0.5, help="Tolerance (µm) to flag touches_border at frame edges"
    ),
):
    """Load polygon boundaries and compute metrics for nuclei/cells.

    Default behavior: scope=both and write a wide per-cell table with cell_* and nuclear_* columns,
    ordered to match the cells table row order when available.
    """
    scope = scope.lower()
    scopes = ["nuclei", "cells"] if scope == "both" else [scope]

    # Load boundaries
    df_cells = read_boundaries(xenium_dir, scope="cells") if "cells" in scopes else None
    df_nucs = (
        read_boundaries(xenium_dir, scope="nuclei") if "nuclei" in scopes else None
    )

    # Frame size for touches_border
    frame = load_frame_um(xenium_dir)
    if frame is None:
        frame = frame_from_extents([d for d in [df_cells, df_nucs] if d is not None])

    frames = []
    if df_cells is not None:
        mc = compute_metrics(
            df_cells, include=metrics, frame_size=frame, border_eps_um=border_eps_um
        )
        frames.append(("cells", mc))
    if df_nucs is not None:
        mn = compute_metrics(
            df_nucs, include=metrics, frame_size=frame, border_eps_um=border_eps_um
        )
        frames.append(("nuclei", mn))

    # Decide output path
    if out_path is None:
        out_dir = xenium_dir / "morphometrics"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "metrics.parquet"
    else:
        out_path.parent.mkdir(parents=True, exist_ok=True)

    # Cells only or Nuclei only → tidy with clear id name and prefixes
    if scope in {"cells", "nuclei"}:
        df = dict(frames)[scope]
        if scope == "cells":
            out = _prefix_columns(df, "cell_", "cell_id")
        else:
            out = _prefix_columns(df, "nuclear_", "nucleus_id")
        _write_table(out, out_path)
        typer.echo(f"Wrote {len(out)} objects → {out_path}")
        return

    # scope == both → wide per-cell table with canonical cell order when available
    cells_tbl = read_cells_table(xenium_dir)  # may be None

    # Prefix metric columns and rename IDs
    cell_metrics = (
        _prefix_columns(dict(frames)["cells"], "cell_", "cell_id")
        if df_cells is not None
        else None
    )
    nuc_metrics = (
        _prefix_columns(dict(frames)["nuclei"], "nuclear_", "nucleus_id")
        if df_nucs is not None
        else None
    )

    if cells_tbl is not None:
        # Preserve canonical order from cells table
        out = cells_tbl[
            [c for c in ["cell_id", "nucleus_id"] if c in cells_tbl.columns]
        ].copy()
        if cell_metrics is not None:
            out = out.merge(cell_metrics, on="cell_id", how="left")
        if nuc_metrics is not None and "nucleus_id" in out.columns:
            out = out.merge(nuc_metrics, on="nucleus_id", how="left")
        elif nuc_metrics is not None:
            # No nucleus mapping column — assume 1:1 id equality
            out = out.merge(
                nuc_metrics.rename(columns={"nucleus_id": "cell_id"}),
                on="cell_id",
                how="left",
            )
    else:
        # No cells table; join on id equality if possible
        if (cell_metrics is not None) and (nuc_metrics is not None):
            out = cell_metrics.merge(
                nuc_metrics.rename(columns={"nucleus_id": "cell_id"}),
                on="cell_id",
                how="outer",
            )
        else:
            out = cell_metrics or nuc_metrics

    # In wide output, scope is redundant; drop any scope columns from merges
    out = out.drop(
        columns=[c for c in out.columns if c.startswith("scope")], errors="ignore"
    )

    _write_table(out, out_path)
    typer.echo(f"Wrote {len(out)} rows → {out_path}")


def _write_table(df: pd.DataFrame, out_path: Path) -> None:
    if out_path.suffix.lower() == ".csv":
        df.to_csv(out_path, index=False)
    else:
        df.to_parquet(out_path, index=False)
