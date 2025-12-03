# H&M Fashion RecSys (Personal Fork)

Lightweight notes for my personal use. Based on the Kaggle H&M Personalized Fashion Recommendations dataset; includes the original recall + LGBM pipeline with an added CatBoost ranker for easy ensembling.

## Project layout
```
data/
  raw/           # place Kaggle CSVs here (articles.csv, customers.csv, transactions_train.csv)
  processed/     # parquet outputs after preprocessing and feature merge
  external/      # optional pretrained embeddings (.npy)
src/
  data/          # preprocessing helpers (DataHelper)
  features/      # feature engineering utils
  retrieval/     # candidate generation rules + collector
  models/        # catboost_ranker.py (added)
```

## Quick start
1) Put Kaggle CSVs into `data/raw/`. (No data is stored in this repo.)
2) Install deps (add others as needed):  
   `pip install -q catboost implicit lightgbm`
3) Preprocess (ID encoding, base features, parquet):  
   ```python
   from src.data.datahelper import DataHelper
   helper = DataHelper(data_dir="data")
   helper.preprocess_data(save=True, name="encoded_full")
   ```
4) Generate candidates + features (use existing notebooks/pipeline; outputs parquet with customer_id, article_id, label, and feature columns).
5) Train CatBoost ranker (optional ensemble with LGBM):  
   ```
   python -m src.models.catboost_ranker \
     --train data/processed/encoded_full/train_weekX.pqt \
     --valid data/processed/encoded_full/valid_weekX.pqt \
     --model-path models/catboost_ranker_weekX.cbm
   ```
6) Predict with CatBoost:  
   ```
   python -m src.models.catboost_ranker \
     --predict data/processed/encoded_full/test_candidates.pqt \
     --model-path models/catboost_ranker_weekX.cbm \
     --output models/catboost_ranker_weekX_preds.pqt
   ```
7) Ensemble with existing LGBM predictions by averaging scores on `(customer_id, article_id)` and take top-12 per user.

## Notes
- No trained models or data are included. Keep large artifacts out of version control.
- DNN code from the original solution remains; I am currently using only LGBM + CatBoost for efficiency.
