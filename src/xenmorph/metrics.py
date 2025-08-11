from __future__ import annotations
import pandas as pd


def compute_morphometrics_stub() -> pd.DataFrame:
    """
    Placeholder morphometrics table with an explicit schema so Parquet write
    succeeds even when there are zero rows.
    """
    # Choose sensible dtypes for an empty table
    schema = {
        "cell_id": "Int64",  # nullable integer
        "nucleus_area_um2": "float64",
        "nucleus_perimeter_um": "float64",
        "circularity": "float64",
        "aspect_ratio": "float64",
        "solidity": "float64",
        "irregularity": "float64",
    }
    df = pd.DataFrame({col: pd.Series(dtype=dtype) for col, dtype in schema.items()})
    return df
