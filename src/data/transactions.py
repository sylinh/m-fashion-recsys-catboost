from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def add_week_info(
    inter: pd.DataFrame,
    *,
    last_date: str = "2020-09-29",
    cutoff_date: Optional[str] = None,
    week_col: str = "week",
    valid_col: str = "valid",
) -> pd.DataFrame:
    """Add `week` and `valid` columns used by existing feature utilities.

    - week: integer weeks since `last_date` (same definition as in notebooks)
    - valid: 1 for every row (placeholder for compatibility)
    """
    if "t_dat" not in inter.columns:
        raise KeyError("Expected column 't_dat'")

    df = inter.copy()
    df["t_dat"] = pd.to_datetime(df["t_dat"])
    if cutoff_date is not None:
        df = df[df["t_dat"] < pd.to_datetime(cutoff_date)].copy()

    df[week_col] = (
        (pd.to_datetime(last_date) - df["t_dat"]).dt.days // 7
    ).astype(np.int16)
    df[valid_col] = np.int8(1)
    return df

