from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from ..data import DataHelper
from ..pipeline.recall import RecallConfig, generate_candidates


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate recall candidates via rule-based retrieval.")
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--dataset", type=str, default="encoded_full", help="processed dataset name")
    p.add_argument("--history-end-date", type=str, help="Only use transactions with t_dat < this date (prevents leakage)")
    p.add_argument("--output", type=Path, required=True, help="output parquet path")
    p.add_argument("--topk-method", type=int, default=200)
    p.add_argument("--topk-merge", type=int, default=500)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    dh = DataHelper(str(args.data_dir))
    data = dh.load_data(args.dataset)

    from ..retrieval import rules as retrieval_rules

    inter = data["inter"].copy()
    inter["t_dat"] = pd.to_datetime(inter["t_dat"])
    if args.history_end_date:
        inter = inter[inter["t_dat"] < pd.to_datetime(args.history_end_date)].copy()

    customers = data["user"]["customer_id"].to_numpy(dtype=np.int32)

    rules: List = [
        retrieval_rules.OrderHistoryDecay(inter, days=14, n=50, name="14d"),
        retrieval_rules.ItemPair(inter, name="pair"),
        retrieval_rules.ALS(customers, inter, days=7, n=50, name="als"),
        retrieval_rules.UserGroupTimeHistory(
            data=data,
            customer_list=customers,
            trans_df=inter.merge(data["user"][["customer_id", "age", "user_gender"]], on="customer_id", how="left"),
            cat_cols=["age", "user_gender"],
            n=50,
            name="ug_pop",
            unique=True,
            scale=True,
        ),
        retrieval_rules.TimeHistoryDecay(customers, inter, days=7, n=50, name="pop7d"),
    ]

    config = RecallConfig(topk_per_method=args.topk_method, topk_after_merge=args.topk_merge)
    candidates = generate_candidates(customer_list=customers, rules=rules, config=config)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    candidates.to_parquet(args.output, index=False)


if __name__ == "__main__":
    main()
