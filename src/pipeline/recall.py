from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from ..retrieval.candidates import (
    CandidateMergeConfig,
    add_source_rank,
    merge_candidates,
    standardize_rule_candidates,
    topk_per_customer,
)


@dataclass(frozen=True)
class RecallConfig:
    topk_per_method: int = 200
    topk_after_merge: int = 500
    merge: CandidateMergeConfig = CandidateMergeConfig(keep="best_score", keep_methods=True)


def generate_candidates(
    *,
    customer_list: np.ndarray,
    rules: Sequence,
    config: RecallConfig = RecallConfig(),
) -> pd.DataFrame:
    """Run a list of retrieval rules and merge into a unified candidate table.

    Returns schema:
      (customer_id, article_id, source_method, source_score, source_rank, [source_methods])
    """
    frames: List[pd.DataFrame] = []
    for rule in rules:
        raw = rule.retrieve()
        std = standardize_rule_candidates(raw)
        std = topk_per_customer(std, k=config.topk_per_method, score_col="source_score")
        frames.append(std)

    merged = merge_candidates(frames, config=config.merge)
    merged = topk_per_customer(merged, k=config.topk_after_merge, score_col="source_score")
    merged = add_source_rank(merged, score_col="source_score", group_col="customer_id", ascending=False)
    return merged


def candidates_to_pair_frame(candidates: pd.DataFrame) -> pd.DataFrame:
    """Drop recall metadata to create a pure (customer_id, article_id) pair set."""
    return candidates[["customer_id", "article_id"]].drop_duplicates().reset_index(drop=True)

