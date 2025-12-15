from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import pandas as pd

from ..utils import merge_week_data
from ..data.transactions import add_week_info


@dataclass(frozen=True)
class FeatureBuildConfig:
    last_date: str = "2020-09-29"
    cutoff_date: Optional[str] = "2020-08-19"  # matches notebook default


class FeatureBuilder:
    """Build supervised training data from (user,item) candidates.

    This wraps the existing `src.utils.merge_week_data` logic to produce:
      (customer_id, article_id, label, feature columns...)
    """

    def __init__(self, data: Dict, *, config: FeatureBuildConfig = FeatureBuildConfig()):
        self.data = data
        self.config = config

    def build_week(
        self,
        *,
        week_num: int,
        candidates: pd.DataFrame,
        trans_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Build features for a given `week_num`.

        Parameters
        ----------
        week_num:
            0 for test, 1..N for train/valid weeks (same convention as notebooks).
        candidates:
            Must contain at least (customer_id, article_id). Can include recall columns.
        trans_df:
            Optional transaction dataframe; defaults to `data['inter']`.
        """
        trans = trans_df if trans_df is not None else self.data["inter"]
        trans = add_week_info(
            trans,
            last_date=self.config.last_date,
            cutoff_date=self.config.cutoff_date,
        )
        return merge_week_data(self.data, trans, week_num, candidates)

