# =============================
# xenmorph/io/xenium.py
# =============================
from __future__ import annotations
from pathlib import Path
import json
import warnings
import pandas as pd
import numpy as np
from typing import Literal, Tuple, Optional

try:
    from shapely.geometry import Polygon
except Exception as e:  # pragma: no cover
    raise RuntimeError("shapely>=2.0 is required for polygon metrics") from e

SCOPE = Literal["nuclei", "cells"]


_DEF_FILES = {
    "nuclei": "nucleus_boundaries.parquet",
    "cells": "cell_boundaries.parquet",
}


def _detect_columns(df: pd.DataFrame) -> Tuple[str, str, str]:
    """Return (id_col, x_col, y_col) for common Xenium boundary schemas.
    Official docs: CSV has columns {cell_id|nucleus_id, vertex_x, vertex_y}.
    Parquet mirrors those names; some tools rename id->id or object_id.
    """
    id_candidates = [
        "nucleus_id",
        "cell_id",
        "id",
        "object_id",
    ]
    x_candidates = ["vertex_x", "x", "coord_x"]
    y_candidates = ["vertex_y", "y", "coord_y"]

    id_col = next((c for c in id_candidates if c in df.columns), None)
    x_col = next((c for c in x_candidates if c in df.columns), None)
    y_col = next((c for c in y_candidates if c in df.columns), None)
    if not (id_col and x_col and y_col):
        raise ValueError(
            f"Could not detect required columns in boundaries: have {list(df.columns)[:10]}..."
        )
    return id_col, x_col, y_col


def read_boundaries(
    xenium_dir: Path | str,
    scope: SCOPE = "nuclei",
    parquet_path: Path | str | None = None,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Load Xenium polygon boundaries for nuclei or cells.

    Returns a DataFrame with columns: [object_id, geometry, scope]. Coordinates are
    in microns (per 10x docs).
    """
    base = Path(xenium_dir)
    pq = Path(parquet_path) if parquet_path else base / _DEF_FILES[scope]
    if not pq.exists():
        raise FileNotFoundError(f"Missing boundaries Parquet: {pq}")

    df = pd.read_parquet(pq, columns=columns)
    id_col, x_col, y_col = _detect_columns(df)

    # Ensure sorted by id and vertex order preserved as in file
    df = df[[id_col, x_col, y_col]].copy()
    df.rename(columns={id_col: "object_id", x_col: "x", y_col: "y"}, inplace=True)

    # Build polygons per object_id
    geoms: list[Polygon] = []
    ids: list[int] = []
    for oid, g in df.groupby("object_id", sort=True):
        arr = g[["x", "y"]].to_numpy(dtype=float)
        # Drop duplicate closing vertex if present
        if len(arr) >= 2 and np.allclose(arr[0], arr[-1]):
            arr = arr[:-1]
        if len(arr) < 3:
            continue  # degenerate
        poly = Polygon(arr)
        if not poly.is_valid:
            poly = poly.buffer(0)  # fix minor self-intersections
        if poly.is_empty or not poly.is_valid:
            continue
        geoms.append(poly)
        ids.append(int(oid))

    out = pd.DataFrame({"object_id": ids, "geometry": geoms})
    out["scope"] = scope
    return out


def read_cells_table(
    xenium_dir: Path | str, columns: Optional[list[str]] = None
) -> Optional[pd.DataFrame]:
    """Return cells table if present (order preserved). Looks for cells.parquet then cells.csv.gz.
    Useful for canonical cell order and nucleus→cell mapping when available.
    """
    base = Path(xenium_dir)
    pqp = base / "cells.parquet"
    csv = base / "cells.csv.gz"
    if pqp.exists():
        return pd.read_parquet(pqp, columns=columns)
    if csv.exists():
        return pd.read_csv(csv, usecols=columns)
    return None


def frame_from_extents(dfs: list[pd.DataFrame]) -> tuple[float, float]:
    """Infer frame size (width_um, height_um) from polygon extents across provided DataFrames.
    Fallback when metadata isn’t available; assumes origin ~ (0,0) and max equals frame edge.
    """
    maxx = 0.0
    maxy = 0.0
    for df in dfs:
        if df is None or df.empty:
            continue
        for g in df["geometry"].values:
            bx, by, Bx, By = g.bounds
            if Bx > maxx:
                maxx = Bx
            if By > maxy:
                maxy = By
    return float(maxx), float(maxy)


def load_frame_um(xenium_dir: Path | str) -> Optional[tuple[float, float]]:
    """Best-effort read of frame size in microns.
    Order: run_metadata.json → morphology_focus.ome.tif (OME) → None (caller can fallback to extents).
    """
    base = Path(xenium_dir)
    meta = base / "run_metadata.json"
    if meta.exists():
        try:
            data = json.loads(meta.read_text())
            # Heuristic keys; different software versions may differ
            w = data.get("width_um") or data.get("frame_width_um")
            h = data.get("height_um") or data.get("frame_height_um")
            if w and h:
                return float(w), float(h)
        except Exception:
            pass
    ome = base / "morphology_focus.ome.tif"
    if ome.exists():
        try:
            import tifffile as tiff

            with tiff.TiffFile(str(ome)) as tf:
                series = tf.series[0]
                shape = series.shape  # (Y, X, C) or similar
                # Guess axes; tifffile exposes series.axes e.g. 'YXC'
                axes = getattr(series, "axes", "")
                sy = shape[axes.index("Y")] if "Y" in axes else shape[-2]
                sx = shape[axes.index("X")] if "X" in axes else shape[-1]
                # Physical sizes from OME (may be missing)
                px = getattr(series, "pixels_physical_size_x", None)
                py = getattr(series, "pixels_physical_size_y", None)
                if px and py:
                    return float(sx) * float(px), float(sy) * float(py)
        except Exception as e:  # pragma: no cover
            warnings.warn(f"Could not read OME metadata for frame size: {e}")
    return None
