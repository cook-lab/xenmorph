# =============================
# xenmorph/io/xenium.py
# =============================
from __future__ import annotations
from pathlib import Path
import json
import warnings
import pandas as pd
import numpy as np
from typing import Literal, Optional

try:
    from shapely.geometry import Polygon
except Exception as e:  # pragma: no cover
    raise RuntimeError("shapely>=2.0 is required for polygon metrics") from e

SCOPE = Literal["nuclei", "cells"]


_DEF_FILES = {
    "nuclei": "nucleus_boundaries.parquet",
    "cells": "cell_boundaries.parquet",
}


def _detect_columns_from_names(names: list[str]) -> tuple[str, str, str]:
    id_candidates = ["nucleus_id", "cell_id", "id", "object_id"]
    x_candidates = ["vertex_x", "x", "coord_x"]
    y_candidates = ["vertex_y", "y", "coord_y"]
    id_col = next((c for c in id_candidates if c in names), None)
    x_col = next((c for c in x_candidates if c in names), None)
    y_col = next((c for c in y_candidates if c in names), None)
    if not (id_col and x_col and y_col):
        raise ValueError(f"Could not detect id/x/y in {names[:10]}...")
    return id_col, x_col, y_col


def _detect_columns(df: pd.DataFrame) -> tuple[str, str, str]:
    """Wrapper that reuses the name-based detector on a DataFrame."""
    return _detect_columns_from_names(list(df.columns))


def read_boundaries(
    xenium_dir: Path | str,
    scope: SCOPE = "nuclei",
    parquet_path: Path | str | None = None,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """
    Load Xenium polygon boundaries for nuclei or cells.

    Returns DataFrame: [object_id (str), geometry (Polygon/MultiPolygon), scope].
    Uses a streaming path for large Parquets to keep memory stable.
    """
    base = Path(xenium_dir)
    pq_path = Path(parquet_path) if parquet_path else base / _DEF_FILES[scope]
    if not pq_path.exists():
        raise FileNotFoundError(f"Missing boundaries Parquet: {pq_path}")

    # Try fast path with pandas for small/moderate files
    try:
        import pyarrow.parquet as pa_parquet  # only needed for streaming

        pf = pa_parquet.ParquetFile(str(pq_path))
        num_rows = pf.metadata.num_rows
        names = getattr(pf.schema_arrow, "names", pf.schema.names)
        id_col, x_col, y_col = _detect_columns_from_names(names)
    except Exception:
        # Fallback: pandas read (original behavior)
        df = pd.read_parquet(pq_path, columns=columns)
        id_col, x_col, y_col = _detect_columns(df)
        df = (
            df[[id_col, x_col, y_col]]
            .copy()
            .rename(columns={id_col: "object_id", x_col: "x", y_col: "y"})
        )
        geoms: list[Polygon] = []
        ids: list[str] = []
        for oid, g in df.groupby("object_id", sort=True):
            arr = g[["x", "y"]].to_numpy(dtype=float)
            if len(arr) >= 2 and np.allclose(arr[0], arr[-1]):
                arr = arr[:-1]
            if len(arr) < 3:
                continue
            poly = Polygon(arr)
            if not poly.is_valid:
                poly = poly.buffer(0)
            if poly.is_empty or not poly.is_valid:
                continue
            geoms.append(poly)
            ids.append(str(oid))
        out = pd.DataFrame({"object_id": ids, "geometry": geoms})
        out["scope"] = scope
        return out

    # Decide whether to stream (threshold ~5M vertices)
    STREAM_THRESHOLD = 5_000_000
    if num_rows <= STREAM_THRESHOLD:
        # small enough: use pandas for simplicity
        df = pd.read_parquet(pq_path, columns=[id_col, x_col, y_col])
        df = df.rename(columns={id_col: "object_id", x_col: "x", y_col: "y"})
        geoms: list[Polygon] = []
        ids: list[str] = []
        for oid, g in df.groupby("object_id", sort=True):
            arr = g[["x", "y"]].to_numpy(dtype=float)
            if len(arr) >= 2 and np.allclose(arr[0], arr[-1]):
                arr = arr[:-1]
            if len(arr) < 3:
                continue
            poly = Polygon(arr)
            if not poly.is_valid:
                poly = poly.buffer(0)
            if poly.is_empty or not poly.is_valid:
                continue
            geoms.append(poly)
            ids.append(str(oid))
        out = pd.DataFrame({"object_id": ids, "geometry": geoms})
        out["scope"] = scope
        return out

    # Streaming path: iterate record batches, finalize polygons when ID changes.
    geoms: list[Polygon] = []
    ids: list[str] = []
    seen_ids: set[str] = set()

    cur_id: str | None = None
    cur_pts: list[tuple[float, float]] = []

    def _flush_current():
        nonlocal cur_id, cur_pts
        if cur_id is None or len(cur_pts) < 3:
            cur_id, cur_pts = None, []
            return
        arr = np.asarray(cur_pts, dtype=float)
        # drop duplicate close
        if len(arr) >= 2 and np.allclose(arr[0], arr[-1]):
            arr = arr[:-1]
        if len(arr) >= 3:
            poly = Polygon(arr)
            if not poly.is_valid:
                poly = poly.buffer(0)
            if poly.is_valid and (not poly.is_empty):
                geoms.append(poly)
                ids.append(cur_id)
        cur_id, cur_pts = None, []

    for batch in pf.iter_batches(columns=[id_col, x_col, y_col], batch_size=262_144):
        # use batch.to_pandas() for simplicity; batch size keeps memory stable
        bdf = batch.to_pandas()
        for oid, x, y in zip(
            bdf[id_col].astype(str), bdf[x_col].astype(float), bdf[y_col].astype(float)
        ):
            if cur_id is None:
                cur_id = oid
            if oid != cur_id:
                # Finalize previous polygon
                if cur_id in seen_ids:
                    raise ValueError(
                        "Boundaries Parquet is not grouped by object ID; "
                        "streaming requires contiguous vertices per object."
                    )
                seen_ids.add(cur_id)
                _flush_current()
                cur_id = oid
            cur_pts.append((x, y))

    # Flush last
    if cur_id is not None:
        if cur_id in seen_ids:
            raise ValueError(
                "Boundaries Parquet is not grouped by object ID; "
                "streaming requires contiguous vertices per object."
            )
        _flush_current()

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
        df = pd.read_parquet(pqp)
        if columns:
            cols = [c for c in columns if c in df.columns]
            return df[cols] if cols else df
        return df
    if csv.exists():
        df = pd.read_csv(csv)
        if columns:
            cols = [c for c in columns if c in df.columns]
            return df[cols] if cols else df
        return df
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
