from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from shapely.geometry import Polygon


# ---- Metric helpers ----
def _exterior_perimeter(g) -> float:
    """Exterior-only perimeter for Polygon or MultiPolygon."""
    gt = getattr(g, "geom_type", None)
    if gt == "Polygon":
        return float(g.exterior.length)
    if gt == "MultiPolygon":
        return float(sum(p.exterior.length for p in g.geoms))
    # Fallback: total length
    return float(g.length)


def _count_holes(g) -> int:
    """Total number of interior rings across Polygon(s)."""
    gt = getattr(g, "geom_type", None)
    if gt == "Polygon":
        return len(getattr(g, "interiors", []))
    if gt == "MultiPolygon":
        return int(sum(len(getattr(p, "interiors", [])) for p in g.geoms))
    return 0


def _aspect_ratio_from_mrr(poly: Polygon) -> tuple[float, float, float]:
    """Return (aspect_ratio, elongation, mrr_area).

    Robust to MultiPolygons by operating on the convex hull.
    aspect_ratio = min(w,h)/max(w,h) in (0,1]; elongation = 1/aspect_ratio.
    mrr_area is used for rectangularity.
    """
    try:
        hull = poly.convex_hull  # handles Polygon or MultiPolygon
        # Guard against degenerate hulls to avoid oriented_envelope warnings
        if getattr(hull, "is_empty", False):
            return float("nan"), float("nan"), float("nan")
        if getattr(hull, "geom_type", "") != "Polygon":
            return float("nan"), float("nan"), float("nan")
        if getattr(hull, "area", 0.0) <= 0.0:
            return float("nan"), float("nan"), float("nan")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            mrr = hull.minimum_rotated_rectangle
        coords = list(mrr.exterior.coords)[:-1]
        if len(coords) < 4:
            return float("nan"), float("nan"), float("nan")

        # Edge lengths of the rectangle
        def _dist(a, b):
            return float(np.hypot(a[0] - b[0], a[1] - b[1]))

        edges = [_dist(coords[i], coords[(i + 1) % 4]) for i in range(4)]
        w = min(edges)
        h = max(edges)
        if w <= 0 or h <= 0:
            return float("nan"), float("nan"), float("nan")
        ar = w / h
        el = h / w
        return ar, el, float(w * h)
    except Exception:
        return float("nan"), float("nan"), float("nan")


def _convexity(g) -> float:
    """Convexity ≤1 using exterior perimeters; robust for MultiPolygons.

    For a Polygon: hull_perimeter / exterior_perimeter.
    For a MultiPolygon: area-weighted mean of per-part convexities (each ≤1).
    """
    gt = getattr(g, "geom_type", None)
    if gt == "Polygon":
        per = float(g.exterior.length)
        hull_ext = float(g.convex_hull.exterior.length)
        return hull_ext / per if per > 0 else np.nan
    if gt == "MultiPolygon":
        parts = [p for p in g.geoms if p.area > 0]
        if not parts:
            return np.nan
        vals, wts = [], []
        for p in parts:
            per = float(p.exterior.length)
            hull_ext = float(p.convex_hull.exterior.length)
            v = hull_ext / per if per > 0 else np.nan
            if np.isfinite(v):
                vals.append(v)
                wts.append(p.area)
        if not vals:
            return np.nan
        return float(
            np.average(np.array(vals, dtype=float), weights=np.array(wts, dtype=float))
        )
    # Fallback (shouldn’t hit for proper polygonal types)
    per = _exterior_perimeter(g)
    hull_ext = float(g.convex_hull.exterior.length)
    return hull_ext / per if per > 0 else np.nan


def flag_touches_border(
    bounds: np.ndarray, width_um: float, height_um: float, eps_um: float = 0.5
) -> np.ndarray:
    """Vectorized fast test using polygon bounds (minx,miny,maxx,maxy)."""
    minx, miny, maxx, maxy = bounds.T
    return (
        (minx <= eps_um)
        | (miny <= eps_um)
        | (maxx >= (width_um - eps_um))
        | (maxy >= (height_um - eps_um))
    )


def compute_metrics(
    df: pd.DataFrame,
    include: list[str] | None = None,
    frame_size: tuple[float, float] | None = None,
    border_eps_um: float = 0.5,
) -> pd.DataFrame:
    """Compute morphometrics on a DataFrame with 'geometry'.

    Metrics (polygon P in µm units):
      - area_um2 = area(P)
      - perimeter_um = length(∂P)
      - equivalent_diameter_um = 2*sqrt(area/π)
      - circularity = 4π*area / perimeter² (1=circle)
      - aspect_ratio = min(w,h)/max(w,h) from minimum rotated rectangle (MRR)
      - elongation = max(w,h)/min(w,h) from MRR
      - rectangularity = area / area(MRR)
      - solidity = area / area(convex_hull)
      - convexity = perimeter(convex_hull) / exterior_perimeter  (≤1; 1 if convex)
      - has_holes, num_holes, is_valid
      - touches_border (fast bound-based test if frame_size is provided)
    """
    geometries = df["geometry"].values
    n = len(geometries)

    # Preallocate
    area = np.empty(n, float)
    peri = np.empty(n, float)
    eqd = np.empty(n, float)
    circ = np.empty(n, float)
    arat = np.empty(n, float)
    elong = np.empty(n, float)
    rect = np.empty(n, float)
    soli = np.empty(n, float)
    conv = np.empty(n, float)
    holes = np.empty(n, bool)
    valid = np.empty(n, bool)
    n_holes = np.empty(n, int)
    bnds = np.empty((n, 4), float)

    for i, g in enumerate(geometries):
        a = float(g.area)
        p = float(g.length)
        area[i] = a
        peri[i] = p
        eqd[i] = 2.0 * np.sqrt(a / np.pi) if a > 0 else 0.0
        eps = 1e-12
        circ[i] = (4.0 * np.pi * a) / (max(p, eps) ** 2)
        ar, el, mrr_area = _aspect_ratio_from_mrr(g)
        arat[i] = ar
        elong[i] = el
        rect[i] = (
            (a / mrr_area)
            if mrr_area and np.isfinite(mrr_area) and mrr_area > 0
            else np.nan
        )
        ch = g.convex_hull
        ch_area = float(ch.area)
        soli[i] = a / ch_area if ch_area > 0 else np.nan
        # Convexity (≤1), robust for MultiPolygons
        conv[i] = _convexity(g)
        hcount = _count_holes(g)
        holes[i] = hcount > 0
        valid[i] = bool(g.is_valid)
        n_holes[i] = hcount
        bnds[i, :] = g.bounds

    out = df[["object_id", "scope"]].copy()
    out["area_um2"] = area
    out["perimeter_um"] = peri
    out["equivalent_diameter_um"] = eqd
    out["circularity"] = circ
    out["aspect_ratio"] = arat
    out["elongation"] = elong
    out["rectangularity"] = rect
    out["solidity"] = soli
    out["convexity"] = conv
    out["has_holes"] = holes
    out["is_valid"] = valid
    out["num_holes"] = n_holes

    if frame_size is not None:
        out["touches_border"] = flag_touches_border(
            bnds, frame_size[0], frame_size[1], border_eps_um
        )

    if include:
        keep = {"object_id", "scope", *include}
        cols = [c for c in out.columns if c in keep]
        return out[cols]
    return out
