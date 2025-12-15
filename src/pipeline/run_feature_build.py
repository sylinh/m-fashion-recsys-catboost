from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ..data import DataHelper
from ..features.builder import FeatureBuilder, FeatureBuildConfig


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build supervised features from recall candidates.")
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--dataset", type=str, default="encoded_full", help="processed dataset name")
    p.add_argument("--week", type=int, required=True, help="week number: 0=test, 1..N=train/valid")
    p.add_argument("--candidates", type=Path, required=True, help="candidates parquet (customer_id,article_id,...)")
    p.add_argument("--output", type=Path, required=True, help="output parquet with label+features")
    p.add_argument("--last-date", type=str, default="2020-09-29")
    p.add_argument("--cutoff-date", type=str, default="2020-08-19")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    dh = DataHelper(str(args.data_dir))
    data = dh.load_data(args.dataset)

    cand = pd.read_parquet(args.candidates)
    fb = FeatureBuilder(
        data,
        config=FeatureBuildConfig(last_date=args.last_date, cutoff_date=args.cutoff_date),
    )
    feat = fb.build_week(week_num=args.week, candidates=cand)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    feat.to_parquet(args.output, index=False)


if __name__ == "__main__":
    main()

