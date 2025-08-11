# =============================
# xenmorph/metrics/core.py
# =============================
from __future__ import annotations
import pandas as pd
import numpy as np
from shapely.geometry import Polygon


# ---- Metric helpers ----


def _aspect_ratio_from_mrr(poly: Polygon) -> tuple[float, float, float]:
    """Return (aspect_ratio, elongation, mrr_area).
    aspect_ratio = min(w,h)/max(w,h) in (0,1]; elongation = 1/aspect_ratio.
    mrr_area is used for rectangularity.
    """
    try:
        mrr = poly.minimum_rotated_rectangle
    except Exception:  # Shapely functional form fallback
        from shapely import minimum_rotated_rectangle

        mrr = minimum_rotated_rectangle(poly)
    coords = list(mrr.exterior.coords)[:-1]
    if len(coords) != 4:
        return float("nan"), float("nan"), float("nan")

    def _dist(a, b):
        return float(np.hypot(a[0] - b[0], a[1] - b[1]))

    lengths = [_dist(coords[i], coords[(i + 1) % 4]) for i in range(4)]
    # Two unique side lengths (each appears twice)
    w, h = sorted(lengths)[:2]
    if w <= 0 or h <= 0:
        return float("nan"), float("nan"), float("nan")
    ar = min(w, h) / max(w, h)
    el = max(w, h) / min(w, h)
    return ar, el, float(w * h)


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
        # Convexity (≤1): use exterior perimeters only
        perim_ext = float(g.exterior.length)
        hull_ext = float(ch.exterior.length)
        conv[i] = hull_ext / perim_ext if perim_ext > 0 else np.nan
        holes[i] = len(getattr(g, "interiors", [])) > 0
        valid[i] = bool(g.is_valid)
        n_holes[i] = len(getattr(g, "interiors", []))
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
