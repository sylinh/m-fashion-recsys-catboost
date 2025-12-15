## Run end-to-end on Kaggle

This repo is designed to run locally or on Kaggle. On Kaggle, the simplest path is:

1) Add the competition dataset `h-and-m-personalized-fashion-recommendations` to your notebook.
2) Upload/clone this repo into the notebook (e.g. `git clone ...`).
3) Run `full_pipeline.py` from the repo root.

Example notebook cell:

```bash
python H-M-Fashion-RecSys/kaggle/full_pipeline.py \
  --input-dir /kaggle/input/h-and-m-personalized-fashion-recommendations \
  --work-dir /kaggle/working/hm-recsys \
  --out-submission /kaggle/working/submission.csv
```

Notes:
- `implicit` (ALS/BPR) is optional; the script will skip ALS if `implicit` is not installed.
- LightGBM is required for the ranker/classifier stages.

