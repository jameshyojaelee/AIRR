# AIRR-ML (AIRR) — HPC-first Kaggle workflow

This repo is an HPC/Slurm-friendly codebase for the AIRR-ML Kaggle challenge:

- **Task 1**: repertoire-level immune state prediction (AUC)
- **Task 2**: label-associated sequence identification (Jaccard vs hidden set)

The main philosophy is **Task‑2‑first** (enrichment / ranking), then use those sequences + strong tabular models to boost Task 1, and finally **per-dataset blending / rank aggregation**.

See `CHAT_PROGRESS.md` for a detailed implementation history and delivered artifacts.

## Data layout

- Training: `train_datasets/train_dataset_1 ... train_dataset_8/` (each contains `metadata.csv` + TSV repertoires)
- Test: `test_datasets/test_datasets/test_dataset_*` (TSV repertoires only)

Runtime overrides:
- `AIRR_TRAIN_ROOT=/path/to/train_datasets`
- `AIRR_TEST_ROOT=/path/to/test_datasets/test_datasets`

## Setup

- Conda (typical on HPC): create/activate your `airrml` environment and run from repo root.
- When running scripts directly, prefer `PYTHONPATH=.` so imports resolve: `PYTHONPATH=. python3 ...`

## Core workflows

### Train + build a submission (all datasets)

`scripts/run_experiment.py` is the main entrypoint:

- Example (enrichment): `PYTHONPATH=. python3 scripts/run_experiment.py --config configs/enrichment_run1.json`
- Example (DeepMIL transformer): `PYTHONPATH=. python3 scripts/run_experiment.py --config configs/deepmil_transformer.json`

Outputs are written under `outputs/<run>-<timestamp>-<cpu|gpu>/` (or `AIRR_OUTPUT_ROOT` override), with per-dataset artifacts and `submission.csv`.

### Slurm launchers (recommended on HPC)

- Enrichment (CPU): `sbatch scripts/submit_enrichment_run.sh`
- DeepMIL transformer (GPU array): `sbatch --array=0-2 scripts/submit_deepmil_transformer.sh`
- Unified HPO (GPU array): `sbatch --array=0-4 scripts/submit_hpo_array.sh`
- Contrastive pretraining (GPU): `sbatch scripts/submit_contrastive_pretrain.sh` (84h time limit)
- DeepMIL fine-tune (GPU): `sbatch scripts/submit_deepmil_finetune.sh` (84h time limit)
- Bayesian Enrichment (CPU): `sbatch scripts/submit_enrichment_bayes.sh` (84h time limit)
- GBM Publicness (CPU): `sbatch scripts/submit_gbm_publicness.sh` (84h time limit)

Monitoring:
- `python3 scripts/monitor_progress.py`
- `squeue -u $USER`, `tail -f logs/<job>.out`

## Models

Models live under `airrml/models/` and are selected via `model_name` in configs.

- `enrichment` (sequence-consuming, Task‑2-first): `airrml/models/enrichment.py`
  - Presence/absence enrichment scoring → ranked sequences for Task 2
  - Repertoire evidence features (sum/max/hits) + logistic calibrator for Task 1
- `deep_mil` (sequence-consuming, GPU): `airrml/models/deep_mil.py`
  - Transformer encoder + attention pooling MIL
  - Used for Task 1 predictions and sequence scoring (attention/gradient-based)
- `kmer_logreg` (tabular, fast baseline): `airrml/models/kmer_logreg.py`
  - k-mer features + logistic regression; can provide sequence importance via projection.
  - **K-mer Lasso** (Strategy #2): A specific L1-regularized config (`configs/kmer_lasso.json`) designed to mine short binding motifs (3-4aa) rather than full sequences.
- `gbm` (tabular): `airrml/models/gradient_boosting.py`
  - Gradient boosting over engineered features
- `enrichment_bayes` (sequence-consuming, Phase 5): `airrml/models/enrichment_bayes.py`
  - Bayesian Beta-Binomial shrinkage for robust sequence scoring (replaces simple log-odds).
- `stacked_ensemble`: `airrml/models/stacked_ensemble.py`
  - Simple ensemble wrapper (OOF stacking)
- `tcrdist_knn`: `airrml/models/tcrdist_knn.py`
  - Lightweight similarity-style model (experimental)

## Feature engineering

Implemented in `airrml/features.py`:
- multi‑k TF–IDF (`k_list`) with optional hashing
- V/J usage and length features
- `GlobalSequencePublicness` (in `airrml/features.py`): Sequence prevalence features across all datasets.
- optional “publicness” block (dataset-dependent; can be memory heavy on ds7/8)

## Submission building + must-pass validation

- Submission builder: `airrml/submission.py` (called by `scripts/run_experiment.py`)
- Validator (use before every Kaggle submit):
  - `PYTHONPATH=. python3 scripts/check_submission.py --submission outputs/.../submission.csv --train-root train_datasets --test-root test_datasets/test_datasets`

## Hybrid and ensembling (recommended)

Hybrid submission (Task 1 from one model, Task 2 from another):
- `python3 scripts/build_hybrid_submission.py --task1-submission <task1.csv> --task2-submission <task2.csv> --out outputs/<run>/submission.csv`

Schema-correct ensemble (Task 1 weighted average + Task 2 rank aggregation):
- `python3 scripts/ensemble_submissions.py --submissions s1.csv s2.csv ... --weights-task1 ... --weights-task2 ... --output outputs/<run>/submission.csv`
- Task 2 methods: `--task2-method rrf|borda|quantile` (default `rrf`)
  - `quantile`: Normalizes ranks to 0-1 scores before averaging (strongest method).
- Optional per-dataset mixing: `--strategy configs/ensemble_strategy_template.json`

Auto-ensemble latest successful runs under `outputs/`:
- `python3 scripts/run_auto_ensemble.py --dedup-near`

Task‑2 stability proxy (choose what to aggregate):
- `python3 scripts/task2_stability.py --submissions s1.csv s2.csv ... --top-n 10000`

## Hyperparameter optimization (HPO)

- Tabular (multi‑k TF–IDF) Optuna tuner: `scripts/tune_optuna.py` + `configs/hpo_multik_tfidf.json`
- Unified HPO harness (Optuna): `scripts/hpo_sweep.py` + `configs/hpo_unified.json` (Slurm array: `scripts/submit_hpo_array.sh`)
- Enrichment Task‑2 sweep (stability proxy): `scripts/enrichment_task2_sweep.py` + `configs/enrichment_sweep_task2.json`

## Notes

- `outputs/`, `train_datasets/`, and `test_datasets/` are gitignored by design.
- For the current “attempt 0.90+” plan, see `project_overview.md`.
