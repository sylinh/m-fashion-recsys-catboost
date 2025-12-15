from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ..ensemble import WeightedEnsembler
from ..retrieval.candidates import to_prediction_strings


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ensemble model scores and generate top-12 predictions.")
    p.add_argument("--ranker", type=Path, help="ranker predictions parquet (customer_id,article_id,score)")
    p.add_argument("--clf", type=Path, help="classifier predictions parquet (customer_id,article_id,score)")
    p.add_argument("--dnn", type=Path, help="dnn predictions parquet (customer_id,article_id,score)")
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument("--out-scores", type=Path, required=True, help="output blended scores parquet")
    p.add_argument("--out-top12", type=Path, required=True, help="output per-user top12 parquet")
    return p.parse_args()


def _load_if(path: Path | None) -> pd.DataFrame | None:
    if path is None:
        return None
    return pd.read_parquet(path)


def main() -> None:
    args = _parse_args()
    preds = {}
    if args.ranker:
        preds["rank"] = pd.read_parquet(args.ranker)
    if args.clf:
        preds["clf"] = pd.read_parquet(args.clf)
    if args.dnn:
        preds["dnn"] = pd.read_parquet(args.dnn)

    ens = WeightedEnsembler(weights={"rank": args.alpha, "clf": args.beta, "dnn": args.gamma})
    blended = ens.blend(preds)

    args.out_scores.parent.mkdir(parents=True, exist_ok=True)
    blended[["customer_id", "article_id", "score"]].to_parquet(args.out_scores, index=False)

    top12 = to_prediction_strings(blended, k=12, score_col="score")
    args.out_top12.parent.mkdir(parents=True, exist_ok=True)
    top12.to_parquet(args.out_top12, index=False)


if __name__ == "__main__":
    main()

