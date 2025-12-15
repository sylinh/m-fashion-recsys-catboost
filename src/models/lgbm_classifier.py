"""LightGBM binary classifier trainer/predictor.

This module expects parquet files containing:
  - customer_id, article_id
  - label (required for train/valid)
  - feature columns
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


ID_COLS = ["customer_id", "article_id"]
LABEL_COL = "label"


def _require_lightgbm():
    try:
        import lightgbm as lgb  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "lightgbm is required. Install with: pip install lightgbm"
        ) from e
    return lgb


def _load_df(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def _infer_feature_cols(df: pd.DataFrame, drop: Optional[List[str]] = None) -> List[str]:
    drop_set = set(ID_COLS + [LABEL_COL])
    if drop:
        drop_set |= set(drop)
    return [c for c in df.columns if c not in drop_set]


def _save_feature_meta(model_path: Path, features: List[str], cat_features: List[str]) -> None:
    meta = {"features": features, "categorical_features": cat_features}
    meta_path = model_path.with_suffix(model_path.suffix + ".features.json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def _load_feature_meta(model_path: Path) -> Tuple[List[str], List[str]]:
    meta_path = model_path.with_suffix(model_path.suffix + ".features.json")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return list(meta["features"]), list(meta.get("categorical_features", []))


def train(
    train_path: Path,
    valid_path: Path,
    model_path: Path,
    *,
    features: Optional[List[str]] = None,
    cat_features: Optional[List[str]] = None,
    num_boost_round: int = 500,
    early_stopping_rounds: int = 50,
    params: Optional[Dict] = None,
) -> None:
    lgb = _require_lightgbm()

    train_df = _load_df(train_path)
    valid_df = _load_df(valid_path)

    if features is None:
        features = _infer_feature_cols(train_df)
    cat_features = cat_features or []

    train_set = lgb.Dataset(
        data=train_df[features],
        label=train_df[LABEL_COL],
        feature_name=features,
        categorical_feature=[c for c in cat_features if c in features],
        free_raw_data=False,
    )
    valid_set = lgb.Dataset(
        data=valid_df[features],
        label=valid_df[LABEL_COL],
        feature_name=features,
        categorical_feature=[c for c in cat_features if c in features],
        free_raw_data=False,
    )

    default_params: Dict = {
        "objective": "binary",
        "boosting_type": "gbdt",
        "metric": "auc",
        "learning_rate": 0.05,
        "num_leaves": 127,
        "max_depth": -1,
        "min_data_in_leaf": 20,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 1,
        "verbosity": -1,
        "seed": 42,
    }
    if params:
        default_params.update(params)

    booster = lgb.train(
        default_params,
        train_set,
        num_boost_round=num_boost_round,
        valid_sets=[valid_set],
        callbacks=[
            lgb.early_stopping(early_stopping_rounds, verbose=True),
            lgb.log_evaluation(period=50),
        ],
    )

    model_path.parent.mkdir(parents=True, exist_ok=True)
    booster.save_model(model_path, num_iteration=booster.best_iteration)
    _save_feature_meta(model_path, features, cat_features)


def predict(
    model_path: Path,
    data_path: Path,
    output_path: Path,
    *,
    features: Optional[List[str]] = None,
) -> None:
    lgb = _require_lightgbm()

    df = _load_df(data_path)
    if features is None:
        features, _ = _load_feature_meta(model_path)

    booster = lgb.Booster(model_file=str(model_path))
    score = booster.predict(df[features], num_iteration=booster.best_iteration)

    out = df[ID_COLS].copy()
    out["score"] = score.astype(np.float32)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output_path, index=False)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train/predict LightGBM binary classifier.")
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--train", dest="train_path", type=Path, help="Train parquet")
    mode.add_argument("--predict", dest="predict_path", type=Path, help="Predict parquet")

    p.add_argument("--valid", dest="valid_path", type=Path, help="Valid parquet (train mode)")
    p.add_argument("--model-path", type=Path, required=True, help="Model file path")
    p.add_argument("--output", dest="output_path", type=Path, help="Prediction parquet output")
    p.add_argument("--cat-features", type=str, help="Comma-separated categorical feature names")
    p.add_argument("--num-boost-round", type=int, default=500)
    p.add_argument("--early-stopping-rounds", type=int, default=50)
    p.add_argument("--params-json", type=str, help="LightGBM params as JSON string")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cat_features = (
        [x.strip() for x in args.cat_features.split(",") if x.strip()]
        if args.cat_features
        else None
    )
    params = json.loads(args.params_json) if args.params_json else None

    if args.train_path:
        if args.valid_path is None:
            raise ValueError("--valid is required in train mode")
        train(
            train_path=args.train_path,
            valid_path=args.valid_path,
            model_path=args.model_path,
            cat_features=cat_features,
            num_boost_round=args.num_boost_round,
            early_stopping_rounds=args.early_stopping_rounds,
            params=params,
        )
        return

    if args.output_path is None:
        raise ValueError("--output is required in predict mode")
    predict(
        model_path=args.model_path,
        data_path=args.predict_path,
        output_path=args.output_path,
    )


if __name__ == "__main__":
    main()

