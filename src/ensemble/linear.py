from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional

import numpy as np
import pandas as pd


ID_COLS = ("customer_id", "article_id")


def _require_cols(df: pd.DataFrame, cols) -> None:
    missing = set(cols) - set(df.columns)
    if missing:
        raise KeyError(f"Missing columns: {sorted(missing)}")


def weighted_sum_ensemble(
    predictions: Mapping[str, pd.DataFrame],
    *,
    weights: Mapping[str, float],
    score_col: str = "score",
    out_score_col: str = "score",
) -> pd.DataFrame:
    """Blend multiple per-(user,item) score frames via weighted sum.

    Each prediction frame must contain: customer_id, article_id, score_col.
    Output contains: customer_id, article_id, out_score_col plus per-model columns.
    """
    if not predictions:
        raise ValueError("predictions is empty")

    merged: Optional[pd.DataFrame] = None
    used = []
    for name, df in predictions.items():
        _require_cols(df, [*ID_COLS, score_col])
        w = float(weights.get(name, 0.0))
        used.append((name, w))

        tmp = df[[*ID_COLS, score_col]].copy()
        tmp = tmp.rename(columns={score_col: f"score_{name}"})
        tmp[f"score_{name}"] = tmp[f"score_{name}"].astype(np.float32, copy=False)

        merged = tmp if merged is None else merged.merge(tmp, on=list(ID_COLS), how="outer")

    assert merged is not None
    for name, _ in used:
        col = f"score_{name}"
        if col not in merged.columns:
            merged[col] = 0.0
        merged[col] = merged[col].fillna(0.0).astype(np.float32, copy=False)

    merged[out_score_col] = 0.0
    for name, w in used:
        if w == 0.0:
            continue
        merged[out_score_col] += w * merged[f"score_{name}"]

    merged[out_score_col] = merged[out_score_col].astype(np.float32, copy=False)
    return merged


@dataclass(frozen=True)
class WeightedEnsembler:
    weights: Dict[str, float]
    score_col: str = "score"

    def blend(self, predictions: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        return weighted_sum_ensemble(
            predictions, weights=self.weights, score_col=self.score_col
        )

