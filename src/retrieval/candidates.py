from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd


REQUIRED_USER_ITEM_COLS = ("customer_id", "article_id")


def standardize_rule_candidates(
    df: pd.DataFrame,
    *,
    method_col: str = "method",
    score_col: str = "score",
    customer_col: str = "customer_id",
    item_col: str = "article_id",
) -> pd.DataFrame:
    """Normalize recall outputs to the project candidate schema.

    Input schema (existing rules): (customer_id, article_id, method, score, ...)
    Output schema: (customer_id, article_id, source_method, source_score)
    """
    missing = {customer_col, item_col, method_col, score_col} - set(df.columns)
    if missing:
        raise KeyError(f"Missing columns: {sorted(missing)}")

    out = df[[customer_col, item_col, method_col, score_col]].copy()
    out = out.rename(
        columns={
            customer_col: "customer_id",
            item_col: "article_id",
            method_col: "source_method",
            score_col: "source_score",
        }
    )
    out["customer_id"] = out["customer_id"].astype(np.int32, copy=False)
    out["article_id"] = out["article_id"].astype(np.int32, copy=False)
    out["source_method"] = out["source_method"].astype("string")
    out["source_score"] = out["source_score"].astype(np.float32, copy=False)
    return out


def add_source_rank(
    candidates: pd.DataFrame,
    *,
    score_col: str = "source_score",
    group_col: str = "customer_id",
    ascending: bool = False,
    rank_col: str = "source_rank",
) -> pd.DataFrame:
    """Add per-user rank based on a score column (1 = best by default)."""
    if group_col not in candidates.columns or score_col not in candidates.columns:
        raise KeyError(f"Expected columns {group_col!r} and {score_col!r}")
    out = candidates.copy()
    out[rank_col] = (
        out.groupby(group_col)[score_col]
        .rank(ascending=ascending, method="first")
        .astype(np.int32)
    )
    return out


@dataclass(frozen=True)
class CandidateMergeConfig:
    keep: str = "best_score"  # best_score | sum_scores
    keep_methods: bool = True
    methods_col: str = "source_methods"


def merge_candidates(
    frames: Sequence[pd.DataFrame],
    *,
    config: CandidateMergeConfig = CandidateMergeConfig(),
) -> pd.DataFrame:
    """Union recall candidates and deduplicate on (customer_id, article_id).

    Expected per-frame schema: (customer_id, article_id, source_method, source_score)
    Output schema:
      - customer_id, article_id
      - source_method: method that wins the aggregation (or "ensemble")
      - source_score: aggregated score
      - source_methods: list[str] of contributing methods (optional)
    """
    if not frames:
        return pd.DataFrame(
            columns=["customer_id", "article_id", "source_method", "source_score"]
        )

    candidates = pd.concat(frames, ignore_index=True)
    missing = set(REQUIRED_USER_ITEM_COLS) - set(candidates.columns)
    if missing:
        raise KeyError(f"Missing columns: {sorted(missing)}")

    required = {"source_method", "source_score"}
    missing2 = required - set(candidates.columns)
    if missing2:
        raise KeyError(f"Missing columns: {sorted(missing2)}")

    candidates = candidates.copy()
    candidates["source_method"] = candidates["source_method"].astype("string")
    candidates["source_score"] = candidates["source_score"].astype(np.float32, copy=False)

    if config.keep == "best_score":
        idx = (
            candidates.sort_values("source_score", ascending=False)
            .groupby(list(REQUIRED_USER_ITEM_COLS), sort=False)
            .head(1)
            .index
        )
        best = candidates.loc[idx, ["customer_id", "article_id", "source_method", "source_score"]]
        if not config.keep_methods:
            return best.reset_index(drop=True)

        methods = (
            candidates.groupby(list(REQUIRED_USER_ITEM_COLS), sort=False)["source_method"]
            .apply(lambda s: list(pd.unique(s.dropna())))
            .reset_index(name=config.methods_col)
        )
        out = best.merge(methods, on=list(REQUIRED_USER_ITEM_COLS), how="left")
        return out.reset_index(drop=True)

    if config.keep == "sum_scores":
        agg = (
            candidates.groupby(list(REQUIRED_USER_ITEM_COLS), sort=False)["source_score"]
            .sum()
            .reset_index()
        )
        out = agg
        out["source_method"] = "ensemble"
        if config.keep_methods:
            methods = (
                candidates.groupby(list(REQUIRED_USER_ITEM_COLS), sort=False)["source_method"]
                .apply(lambda s: list(pd.unique(s.dropna())))
                .reset_index(name=config.methods_col)
            )
            out = out.merge(methods, on=list(REQUIRED_USER_ITEM_COLS), how="left")
        return out[["customer_id", "article_id", "source_method", "source_score", *([config.methods_col] if config.keep_methods else [])]]

    raise ValueError(f"Unsupported keep={config.keep!r}")


def topk_per_customer(
    candidates: pd.DataFrame,
    *,
    k: int = 200,
    score_col: str = "source_score",
) -> pd.DataFrame:
    if k <= 0:
        raise ValueError("k must be > 0")
    if "customer_id" not in candidates.columns or score_col not in candidates.columns:
        raise KeyError("Expected columns 'customer_id' and a score column")

    out = candidates.sort_values(["customer_id", score_col], ascending=[True, False])
    out = out.groupby("customer_id", sort=False).head(k)
    return out.reset_index(drop=True)


def to_prediction_strings(
    ranked: pd.DataFrame,
    *,
    k: int = 12,
    item_col: str = "article_id",
    customer_col: str = "customer_id",
    score_col: str = "score",
    prediction_col: str = "prediction",
) -> pd.DataFrame:
    """Convert per-(user,item) scores into Kaggle submission format."""
    required = {customer_col, item_col, score_col}
    missing = required - set(ranked.columns)
    if missing:
        raise KeyError(f"Missing columns: {sorted(missing)}")

    df = ranked[[customer_col, item_col, score_col]].copy()
    df = df.sort_values([customer_col, score_col], ascending=[True, False])
    df = df.drop_duplicates([customer_col, item_col], keep="first")
    df = df.groupby(customer_col, sort=False)[item_col].apply(lambda s: " ".join(map(str, s.head(k).tolist()))).reset_index()
    df = df.rename(columns={item_col: prediction_col, customer_col: "customer_id"})
    return df

