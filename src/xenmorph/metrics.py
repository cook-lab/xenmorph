from __future__ import annotations
import pandas as pd


def compute_morphometrics_stub() -> pd.DataFrame:
    """
    Temporary placeholder. Returns an empty DataFrame with the columns we expect
    to fill later. This makes downstream writing/typing predictable.
    """
    cols = [
        "cell_id",
        "nucleus_area_um2",
        "nucleus_perimeter_um",
        "circularity",
        "aspect_ratio",
        "solidity",
        "irregularity",
    ]
    return pd.DataFrame(columns=cols)
