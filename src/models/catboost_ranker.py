"""CatBoost ranker utility to reuse existing recall candidates/features.

Usage examples
--------------
Train:
    python -m src.models.catboost_ranker \
        --train data/processed/encoded_full/train_week3.pqt \
        --valid data/processed/encoded_full/valid_week3.pqt \
        --model-path models/catboost_ranker_week3.cbm

Predict:
    python -m src.models.catboost_ranker \
        --predict data/processed/encoded_full/test_candidates.pqt \
        --model-path models/catboost_ranker_week3.cbm \
        --output models/catboost_ranker_week3_preds.pqt

The script expects parquet files that already contain:
    - customer_id, article_id
    - label (optional for predict)
    - feature columns engineered in notebooks/merge_week_data
It groups by customer_id for ranking. CatBoost must be installed.
"""

import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
from catboost import CatBoostRanker, Pool


ID_COLS = ["customer_id", "article_id"]
LABEL_COL = "label"


def _load_df(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    return df


def _split_features(
    df: pd.DataFrame, cat_features: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, Optional[pd.Series], List[int]]:
    feature_cols = [c for c in df.columns if c not in ID_COLS + [LABEL_COL]]
    X = df[feature_cols]
    y = df[LABEL_COL] if LABEL_COL in df.columns else None

    cat_idx: List[int] = []
    if cat_features:
        cat_idx = [feature_cols.index(c) for c in cat_features if c in feature_cols]

    return X, y, cat_idx


def _make_pool(
    df: pd.DataFrame, cat_features: Optional[List[str]] = None
) -> Pool:
    X, y, cat_idx = _split_features(df, cat_features)
    group_id = df["customer_id"]
    pool = Pool(data=X, label=y, group_id=group_id, cat_features=cat_idx)
    return pool


def train(
    train_path: Path,
    valid_path: Optional[Path],
    model_path: Path,
    iterations: int = 500,
    depth: int = 8,
    learning_rate: float = 0.1,
    loss_function: str = "YetiRank",
    eval_top_k: int = 12,
    cat_features: Optional[List[str]] = None,
    random_seed: int = 42,
) -> None:
    train_df = _load_df(train_path)
    train_pool = _make_pool(train_df, cat_features)

    eval_set = None
    if valid_path is not None:
        valid_df = _load_df(valid_path)
        eval_set = _make_pool(valid_df, cat_features)

    model = CatBoostRanker(
        iterations=iterations,
        depth=depth,
        learning_rate=learning_rate,
        loss_function=loss_function,
        eval_metric=f"NDCG:top={eval_top_k}",
        random_seed=random_seed,
        verbose=100,
        task_type="CPU",
    )
    model.fit(train_pool, eval_set=eval_set, use_best_model=eval_set is not None)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(model_path)


def predict(
    model_path: Path,
    data_path: Path,
    output_path: Path,
    cat_features: Optional[List[str]] = None,
) -> None:
    df = _load_df(data_path)
    pool = _make_pool(df, cat_features)

    model = CatBoostRanker()
    model.load_model(model_path)

    preds = model.predict(pool)

    out = df[ID_COLS].copy()
    out["score"] = preds

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output_path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train or predict CatBoost Ranker.")

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--train", dest="train_path", type=Path, help="Train parquet")
    mode.add_argument("--predict", dest="predict_path", type=Path, help="Predict parquet")

    parser.add_argument("--valid", dest="valid_path", type=Path, help="Valid parquet")
    parser.add_argument("--model-path", type=Path, required=True, help="Path to save/load model")
    parser.add_argument("--output", dest="output_path", type=Path, help="Prediction parquet output")
    parser.add_argument("--cat-features", type=str, help="Comma-separated categorical feature names")
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--loss-function", type=str, default="YetiRank")
    parser.add_argument("--eval-top-k", type=int, default=12)
    parser.add_argument("--random-seed", type=int, default=42)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cat_features = (
        [x.strip() for x in args.cat_features.split(",") if x.strip()]
        if args.cat_features
        else None
    )

    if args.train_path:
        train(
            train_path=args.train_path,
            valid_path=args.valid_path,
            model_path=args.model_path,
            iterations=args.iterations,
            depth=args.depth,
            learning_rate=args.learning_rate,
            loss_function=args.loss_function,
            eval_top_k=args.eval_top_k,
            cat_features=cat_features,
            random_seed=args.random_seed,
        )
    else:
        if args.output_path is None:
            raise ValueError("--output is required in predict mode")
        predict(
            model_path=args.model_path,
            data_path=args.predict_path,
            output_path=args.output_path,
            cat_features=cat_features,
        )


if __name__ == "__main__":
    main()
