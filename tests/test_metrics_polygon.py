# =============================
# tests/test_metrics_polygon.py (minimal synthetic)
# =============================
import pandas as pd
from shapely.geometry import Point

from xenmorph.metrics.core import compute_metrics


def _circle_poly(center=(0, 0), r=10.0, n=256):
    # approximate circle with shapely buffer
    return Point(center).buffer(r, quad_segs=64)


def test_circle_circularity_near_one():
    poly = _circle_poly(r=10)
    df = pd.DataFrame({"object_id": [1], "scope": ["nuclei"], "geometry": [poly]})
    out = compute_metrics(df)
    c = float(out.loc[0, "circularity"])
    assert 0.98 <= c <= 1.02


def test_basic_columns_present():
    poly = _circle_poly(r=5)
    df = pd.DataFrame({"object_id": [42], "scope": ["cells"], "geometry": [poly]})
    out = compute_metrics(df)
    assert set(
        ["area_um2", "perimeter_um", "circularity", "aspect_ratio", "solidity"]
    ).issubset(out.columns)
