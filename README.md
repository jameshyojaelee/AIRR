# AIRR Project 

- **Goal**: Predict immune labels for repertoires and find important sequences for the AIRR-ML Kaggle challenge.
- **Data layout**:
  - `train_datasets/` with subfolders `train_dataset_1 ... 8` (metadata + TSVs).
  - `test_datasets/test_datasets/` with subfolders `test_dataset_*` (TSVs only).
- **Models I try**:
  - `kmer_logreg`: k-mer counts + L1 logistic regression (fast baseline).
  - `gbm`: gradient boosting (uses xgboost/lightgbm/sklearn, whichever is installed).
  - `deep_mil`: deep multiple-instance learning in PyTorch (uses attention over sequences).
- **New HPO utility**: `scripts/tune_optuna.py --config configs/hpo_multik_tfidf.json` runs Optuna CV sweeps (multi-k TF–IDF k=3–6 with optional hashing) for `kmer_logreg` and `gbm`, saving best params and trial logs to `outputs/hpo_multik_tfidf/`.
- **Pipelines**:
  - `scripts/run_experiment.py --config <json>` trains across all train datasets and builds `submission.csv`.
  - Outputs go to `outputs/<run>-<date>-<cpu|gpu>/`, each with per-dataset artifacts and a `run_summary.csv`.
