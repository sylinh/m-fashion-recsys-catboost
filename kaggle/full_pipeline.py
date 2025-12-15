from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run end-to-end H&M pipeline on Kaggle.")
    p.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Kaggle dataset folder (contains articles.csv, customers.csv, transactions_train.csv, sample_submission.csv).",
    )
    p.add_argument(
        "--work-dir",
        type=Path,
        required=True,
        help="Working directory for this repo's expected layout (will create data/raw, data/processed, models).",
    )
    p.add_argument("--dataset-name", type=str, default="encoded_full")
    p.add_argument("--out-submission", type=Path, required=True)
    p.add_argument("--train-weeks", type=str, default="2,3,4,5", help="comma-separated")
    p.add_argument("--valid-week", type=int, default=1)
    p.add_argument("--recall-topk-method", type=int, default=200)
    p.add_argument("--recall-topk-merge", type=int, default=500)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--gamma", type=float, default=0.0, help="DNN weight (not trained by this script)")
    return p.parse_args()


def _calc_valid_date(week_num: int, last_date: str = "2020-09-29") -> tuple[str, str]:
    end_date = pd.to_datetime(last_date) - pd.Timedelta(days=7 * week_num - 1)
    start_date = end_date - pd.Timedelta(days=7)
    return start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")


def _ensure_repo_on_path(repo_root: Path) -> None:
    import sys

    sys.path.insert(0, str(repo_root / "H-M-Fashion-RecSys"))


def _copy_raw(input_dir: Path, raw_dir: Path) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    for name in ["articles.csv", "customers.csv", "transactions_train.csv", "sample_submission.csv"]:
        src = input_dir / name
        if not src.exists():
            if name == "sample_submission.csv":
                continue
            raise FileNotFoundError(f"Missing {src}")
        shutil.copy2(src, raw_dir / name)


def _try_import_implicit() -> bool:
    try:
        import implicit  # type: ignore  # noqa: F401
    except Exception:
        return False
    return True


def _train_concat_paths(work_dir: Path, week_nums: List[int]) -> Path:
    out = work_dir / "data" / "processed" / "pipeline" / f"train_weeks_{'-'.join(map(str, week_nums))}.pqt"
    return out


def main() -> None:
    args = _parse_args()
    repo_root = args.work_dir
    repo_root.mkdir(parents=True, exist_ok=True)

    _ensure_repo_on_path(repo_root)

    from src.data import DataHelper, IdMap
    from src.features import FeatureBuilder, FeatureBuildConfig
    from src.models import lgbm_classifier, lgbm_ranker
    from src.retrieval.candidates import to_prediction_strings
    from src.ensemble import WeightedEnsembler

    data_dir = repo_root / "data"
    raw_dir = data_dir / "raw"
    _copy_raw(args.input_dir, raw_dir)

    dh = DataHelper(str(data_dir))
    processed_dir = data_dir / "processed" / args.dataset_name
    if not processed_dir.exists():
        dh.preprocess_data(save=True, name=args.dataset_name)

    data = dh.load_data(args.dataset_name)
    id_map = IdMap.load(data_dir)

    inter = data["inter"].copy()
    inter["t_dat"] = pd.to_datetime(inter["t_dat"])

    # Customers from sample submission (test set)
    sample_path = raw_dir / "sample_submission.csv"
    if sample_path.exists():
        sample = pd.read_csv(sample_path)
        sample_customer_idx = sample["customer_id"].map(id_map.user_id2index).dropna().astype(np.int32).unique()
        test_customers = sample_customer_idx
    else:
        test_customers = data["user"]["customer_id"].to_numpy(dtype=np.int32)

    # ========================= Recall (week0 test) =========================
    from src.pipeline.recall import RecallConfig, generate_candidates
    from src.retrieval import rules as retrieval_rules

    has_implicit = _try_import_implicit()

    def make_rules(customer_list: np.ndarray, history: pd.DataFrame) -> List:
        rules: List = [
            retrieval_rules.OrderHistoryDecay(history, days=14, n=50, name="14d"),
            retrieval_rules.ItemPair(history, name="pair"),
            retrieval_rules.UserGroupTimeHistory(
                data=data,
                customer_list=customer_list,
                trans_df=history.merge(data["user"][["customer_id", "age", "user_gender"]], on="customer_id", how="left"),
                cat_cols=["age", "user_gender"],
                n=50,
                name="ug_pop",
                unique=True,
                scale=True,
            ),
            retrieval_rules.TimeHistoryDecay(customer_list, history, days=7, n=50, name="pop7d"),
        ]
        if has_implicit:
            rules.insert(2, retrieval_rules.ALS(customer_list, history, days=7, n=50, name="als"))
        return rules

    recall_cfg = RecallConfig(
        topk_per_method=args.recall_topk_method,
        topk_after_merge=args.recall_topk_merge,
    )

    test_rules = make_rules(test_customers, inter)
    test_candidates = generate_candidates(customer_list=test_customers, rules=test_rules, config=recall_cfg)

    # ========================= Feature build =========================
    fb = FeatureBuilder(data, config=FeatureBuildConfig())
    test_feat = fb.build_week(week_num=0, candidates=test_candidates, trans_df=inter)

    # Train/valid from historical weeks (prevent leakage by cutting history at each week end_date).
    train_weeks = [int(x) for x in args.train_weeks.split(",") if x.strip()]
    valid_week = int(args.valid_week)

    def build_week_dataset(w: int) -> pd.DataFrame:
        _, end_date = _calc_valid_date(w)
        hist = inter[inter["t_dat"] < pd.to_datetime(end_date)].copy()
        cust = data["user"]["customer_id"].to_numpy(dtype=np.int32)
        rules = make_rules(cust, hist)
        cand = generate_candidates(customer_list=cust, rules=rules, config=recall_cfg)
        return fb.build_week(week_num=w, candidates=cand, trans_df=hist)

    train_frames = [build_week_dataset(w) for w in train_weeks]
    train_df = pd.concat(train_frames, ignore_index=True)
    valid_df = build_week_dataset(valid_week)

    pipeline_dir = data_dir / "processed" / "pipeline"
    pipeline_dir.mkdir(parents=True, exist_ok=True)
    train_path = _train_concat_paths(repo_root, train_weeks)
    valid_path = pipeline_dir / f"valid_week_{valid_week}.pqt"
    test_path = pipeline_dir / "test_week0.pqt"
    train_df.to_parquet(train_path, index=False)
    valid_df.to_parquet(valid_path, index=False)
    test_feat.to_parquet(test_path, index=False)

    # ========================= Train + predict models =========================
    model_dir = repo_root / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    cat_features = [
        "customer_id",
        "article_id",
        "product_code",
        "FN",
        "Active",
        "club_member_status",
        "fashion_news_frequency",
        "age",
        "product_type_no",
        "product_group_name",
        "graphical_appearance_no",
        "colour_group_code",
        "perceived_colour_value_id",
        "perceived_colour_master_id",
        "user_gender",
        "article_gender",
        "season_type",
    ]

    rank_model = model_dir / "lgbm_ranker.model"
    clf_model = model_dir / "lgbm_classifier.model"

    lgbm_ranker.train(
        train_path=train_path,
        valid_path=valid_path,
        model_path=rank_model,
        cat_features=cat_features,
        params={"learning_rate": 0.03, "num_leaves": 128, "max_depth": 8},
        num_boost_round=300,
        early_stopping_rounds=30,
    )
    lgbm_classifier.train(
        train_path=train_path,
        valid_path=valid_path,
        model_path=clf_model,
        cat_features=cat_features,
        params={"learning_rate": 0.03, "num_leaves": 128, "max_depth": 8},
        num_boost_round=300,
        early_stopping_rounds=30,
    )

    rank_pred_path = model_dir / "test_rank_preds.pqt"
    clf_pred_path = model_dir / "test_clf_preds.pqt"

    lgbm_ranker.predict(rank_model, test_path, rank_pred_path)
    lgbm_classifier.predict(clf_model, test_path, clf_pred_path)

    # ========================= Ensemble + submission =========================
    rank_pred = pd.read_parquet(rank_pred_path)
    clf_pred = pd.read_parquet(clf_pred_path)

    ens = WeightedEnsembler(weights={"rank": args.alpha, "clf": args.beta, "dnn": args.gamma})
    blended = ens.blend({"rank": rank_pred, "clf": clf_pred})

    pred_str = to_prediction_strings(blended, k=12, score_col="score")

    # Map indices back to original IDs for submission
    pred_str["customer_id"] = pred_str["customer_id"].map(id_map.user_index2id)
    pred_str["prediction"] = pred_str["prediction"].apply(
        lambda s: " ".join(str(id_map.item_index2id[int(x)]) for x in s.split()) if isinstance(s, str) and s else ""
    )

    if sample_path.exists():
        sample = pd.read_csv(sample_path)[["customer_id"]]
        out = sample.merge(pred_str, on="customer_id", how="left")
        out["prediction"] = out["prediction"].fillna("")
    else:
        out = pred_str

    args.out_submission.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_submission, index=False)


if __name__ == "__main__":
    main()

