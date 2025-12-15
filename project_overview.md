# AIRR-ML-25: Adaptive Immune Profiling Challenge 2025

Project overview for participating in the AIRR‑ML‑25 Kaggle competition. This document serves as the central context for the challenge details, our codebase architecture, and the execution roadmap.

---

## 1. Challenge summary

AIRR‑ML‑25 is a machine learning challenge on adaptive immune receptor repertoires (AIRR). Participants build models that, for multiple datasets:

1.  Predict the immune state of individuals or samples (e.g., disease vs healthy) from their adaptive immune receptor repertoires.
2.  Identify receptor sequences that are most associated with that immune state.

The overarching goal is to accelerate development of machine learning methods for immunodiagnostics and therapeutics discovery. :contentReference[oaicite:0]{index=0}

---

## 2. Timeline and platform

-   **Platform**
    -   Hosted on Kaggle: `https://www.kaggle.com/competitions/adaptive-immune-profiling-challenge-2025` :contentReference[oaicite:1]{index=1}

-   **Key dates**
    -   Start: **05 November 2025, 08:00 CET**
    -   Final submission deadline: **17 December 2025, 07:59 CET** :contentReference[oaicite:2]{index=2}

All deadlines use Central European Time (CET).

-   **How to join**
    -   Create a Kaggle account.
    -   Accept competition rules.
    -   Submit predictions via `submission.csv`.

---

## 3. Prizes and publication

### 3.1 Monetary prizes
-   **1st place**: 5,000 USD
-   **2nd place**: 3,000 USD
-   **3rd place**: 2,000 USD :contentReference[oaicite:5]{index=5}

Eligibility requires releasing code as open source (MIT license).

### 3.2 Scientific manuscript authorship
-   **Top 10 teams** invoked to contribute to a Nature Methods article. :contentReference[oaicite:8]{index=8}

---

## 4. Formal problem definition

You are given multiple datasets of T cell receptor (TCR) repertoires.

### 4.1 Task 1: Immune state prediction
-   **Input**: Repertoires (sets of receptor sequences) in test datasets.
-   **Output**: Probability that the repertoire is **label positive** (binary classification).
-   **Metric**: Area Under the ROC Curve (AUC).

### 4.2 Task 2: Label associated sequence identification
-   **Input**: Training repertoires with labels.
-   **Output**: Ranked list of up to **50,000 unique sequences** (junction_aa, v_call, j_call) per training dataset.
-   **Metric**: Jaccard similarity against a hidden ground truth set of associated sequences.

---

## 5. Datasets

### 5.1 Number of datasets
-   **Training**: 8 datasets (`train_datasets/train_datasets/train_dataset_X`)
-   **Test**: 10 datasets (`test_datasets/test_datasets/test_dataset_X_Y`)

### 5.2 Data sources
-   **Adaptive Biotechnologies**: ~500 unpublished TCRβ repertoires (HSV-2 infection status).
-   **Parse Biosciences**: ~1M antigen-specific TCR sequences (experiments).
-   Synthetic datasets are constructed from these sources.

---

## 6. Evaluation and leaderboard

-   **Per dataset**:
    -   Task 1: AUC
    -   Task 2: Jaccard Similarity
-   **Overall**:
    -   Weighted average over datasets and metrics.

---

## 7. Expected submission format

Single `submission.csv` containing:
-   **Repertoire Predictions**: `ID` (repertoire_id), `dataset`, `label_positive_probability`.
-   **Sequence Rankings**: `ID` (sequence_rank_id), `dataset`, `junction_aa`, `v_call`, `j_call`.

---

## 8. Codebase Architecture

Our solution is built on a modular Python package `airrml` with a clear separation of concerns.

### 8.1 Directory Structure
-   `airrml/`: Core package.
    -   `models/`: Model implementations.
        -   `deep_mil.py`: **Deep MIL Transformer**. PyTorch-based Multiple Instance Learning model. Encodes sequences with a Transformer, pools them with Attention to predict repertoire label. Attention weights serve as sequence importance scores.
        -   `kmer_logreg.py`: **Baseline**. Logistic Regression on k-mer counts.
        -   `gradient_boosting.py`: XGBoost/LightGBM/CatBoost wrapper.
    -   `features.py`: Feature engineering (k-mer encoding, TF-IDF).
    -   `pipeline.py`: Utilities for training, cross-validation, and prediction.
-   `configs/`: JSON configuration files for experiments.
-   `scripts/`: Executable scripts for workflows.
    -   `run_experiment.py`: Main entry point for full training.
    -   `tune_optuna.py`: Hyperparameter optimization.
    -   `submit_*.sh`: Slurm submission scripts for HPC.

### 8.2 Key Models
1.  **Deep MIL (`deep_mil`)**:
    -   **Architecture**: Embedding -> Transformer Encoder -> Attention Pooling -> Classifier.
    -   **Mechanism**: Learns a representation for each sequence, then learns to attend to "important" sequences to classify the repertoire.
    -   **Task 2**: Uses attention weights (plus optional gradient saliency) to rank sequences.
    
2.  **K-mer Logistic Regression (`kmer_logreg`)**:
    -   **Mechanism**: Counts 3-mers (or k-mers) across the entire repertoire.
    -   **Task 2**: Ranks sequences by summing the learned coefficients of the k-mers they contain.

---

## 9. Current Status & Roadmap

### 9.1 Status (as of Dec 14, 2025)
-   **Implemented**:
    -   End-to-end pipeline for training and Kaggle submission generation.
    -   `deep_mil`: Transformer MIL (configurable depth/heads/dropout) + AMP/grad clipping/early-stop; supports attention-based sequence importance and gradient-assisted scoring.
    -   `kmer_logreg`: k-mer + logistic regression (L1/L2).
    -   `gbm`: Gradient boosting backend (XGBoost/LightGBM/sklearn fallback).
    -   Multi-k TF–IDF features + optional hashing for tabular models.
    -   Contrastive pretraining scaffold (`scripts/run_contrastive_pretrain.py`) to pretrain an encoder and fine-tune DeepMIL.
    -   HPO via Optuna (both per-model and unified sweep harness) + Slurm array launchers.
    -   Submission assembly from existing DeepMIL artifacts (`scripts/submit_assemble_deepmil_outputs.sh`).
    -   Stacking ensemble scaffold + calibration, and a lightweight similarity model (`tcrdist_knn`) as an additional diversity source.
-   **Verified**:
    -   Data loading and dataset discovery.
    -   DeepMIL submissions can be assembled successfully from saved artifacts.
    -   Optuna HPO produces per-dataset trial logs and best params.
    -   Ensembling: `scripts/ensemble_submissions.py` correctly blends Task 1 probabilities and aggregates Task 2 rankings (RRF/Borda/Quantile).
    -   Bayesian Enrichment: `enrichment_bayes` model implemented with Beta-Binomial shrinkage.


### 9.2 Reality Check: What “90%+” Actually Means
The leaderboard score is a weighted blend of:
-   **Task 1 AUC** (repertoire prediction)
-   **Task 2 Jaccard** (sequence set recovery)
aggregated across multiple datasets.

Reaching **0.90+ overall** is only plausible if we can do *both*:
1) near-perfect AUC on most datasets, and  
2) strong Task 2 sequence recovery (Jaccard) on most datasets.

This requires a plan that is **Task-2-first** (sequence discovery), not only a model tuning plan for AUC.

### 9.3 Strategy to Maximize Score (High-Level)
We will pursue three complementary “engines”, then blend them per dataset:

1) **Sequence enrichment engine (highest leverage for Task 2)**  
   A dedicated statistical enrichment method that directly ranks sequences by label association (e.g., Fisher exact / log-odds with priors), plus a repertoire scoring rule built from the same enriched sequences.
   -   This often dominates when datasets contain synthetic spikes or antigen-specific motifs.
   -   Produces high-quality ranked lists for Task 2 and strong Task 1 signals.

2) **DeepMIL engine (non-linear motif learning)**  
   Transformer MIL with aggressive scaling (sweeps over depth/heads, length, subsampling, augmentations, contrastive pretrain).
   -   Valuable for datasets with distributed signals not captured by simple enrichment.

3) **Tabular engine (robust baseline + calibration)**  
   Multi-k TF–IDF + logreg, tuned GBM, plus lightweight similarity models.
   -   Often provides strong “publicness” features and stable generalization.

Final submissions should be a **per-dataset calibrated blend** of these engines for Task 1, and a **rank-aggregation blend** for Task 2.

### 9.4 Execution Plan (Dec 14 – Dec 17)
This is the practical plan to *attempt* 0.90+ with HPC scale and a 5-submissions/day constraint.

#### Phase 0 — Measurement, Guardrails, and Submission Budget (Today, <3 hours)
-   [ ] Establish a single “experiment ledger” (CSV/SQLite) with: config hash, git hash, dataset, CV AUC, runtime, produced submission path, Kaggle score.
-   [ ] Decide a submission budget: **2 exploratory submissions/day**, **3 confirmatory submissions/day**.
-   [ ] Define “must-pass” checks before a Kaggle submission:
    -   Submission file schema + row counts
    -   No NaNs, no missing columns
    -   Deterministic rerun on toy data

#### Phase 1 — Build the Sequence Enrichment Engine (Today → Tomorrow)
Goal: make Task 2 competitive and convert it into Task 1 features.

Deliverables:
-   New model `airrml/models/enrichment.py` (or similar) that outputs:
    -   `sequence_importance` ranked list (junction_aa, v_call, j_call, score)
    -   repertoire prediction probabilities based on enriched-sequence evidence
-   New config(s) + Slurm script to run it across all datasets.

Method outline:
-   For each dataset:
    1) Create a binary presence/absence table for unique (junction_aa, v_call, j_call).
    2) Compute enrichment per sequence:
       - Fisher exact or log-odds ratio with pseudocount (empirical Bayes prior).
       - Use effect size + p-value shrinkage (avoid over-ranking ultra-rare sequences).
    3) Rank sequences by a combined score, e.g.:
       - `score = signed_log_odds * sqrt(min(pos_count, neg_count))` or a similar stabilized statistic.
    4) Task 2 list: top 50k after dedup + near-duplicate collapse.
    5) Task 1 repertoire score:
       - Sum top-N enriched sequence indicators (or weighted counts),
       - then calibrate via logistic regression on training repertoires.

Quality controls (important):
-   Verify stability of the ranked list across CV folds (proxy for Jaccard robustness):
    -   Compute fold-to-fold Jaccard of top-10k lists (we don’t know ground truth, but stability matters).
    -   Prefer models with high stability *and* good AUC.

#### Phase 2 — DeepMIL Scaling With Contrastive Pretraining (Tomorrow)
Goal: pick 1–2 DeepMIL configurations that consistently help across datasets.

Work items:
-   [ ] Contrastive pretrain on pooled sequences (all datasets), save encoder, fine-tune per dataset.
-   [ ] Increase capacity: max_len 60–90, model_dim 384–768, 4–8 layers, dropout 0.1–0.3.
-   [ ] Add heavy augmentation toggles per dataset (drop/mask/shuffle) and subsampling sweeps.
-   [ ] Run LOCO validation for generalization estimation (Task 1).
-   [ ] For Task 2: use fold/seed consensus ranking:
    -   Train multiple seeds, export per-seed ranked lists, aggregate ranks (reciprocal-rank fusion).

#### Phase 3 — Tabular Scaling + Publicness Controls (Tomorrow → Next Day)
Goal: create strong, stable signals and good calibration to blend with deep models.

Work items:
-   [ ] Multi-k TF–IDF (k=3–6) + L2 logreg + calibration.
-   [ ] GBM tuned per dataset; include V/J usage and length stats.
-   [ ] Add “publicness” features:
    -   Per sequence: frequency across repertoires (not just counts within repertoire).
    -   Downweight overly public sequences (common across both labels).

#### Phase 4 — Ensembling & Post-processing (Next Day)
Goal: combine engines intelligently rather than averaging.

Task 1:
-   [ ] Per-dataset stacking/blending:
    -   OOF predictions for each engine (enrichment, deep, tabular).
    -   Train a blender (logistic regression) per dataset.
    -   Calibrate final probabilities (isotonic/sigmoid) per dataset.

Task 2:
-   [ ] Rank aggregation (per dataset):
    -   Combine enriched list + DeepMIL list + k-mer list using reciprocal-rank fusion.
    -   Enforce uniqueness, collapse near-duplicates, and ensure top portion is high-confidence.

#### Phase 5 — Optimization & Refinement (Dec 15)
Goal: robustness and fine-tuning.

Work items:
- [x] **Bayesian Enrichment**: Replace brittle log-odds filters with Beta-Binomial estimates (`airrml/models/enrichment_bayes.py`).
- [ ] **Quantile Ensembling**: Normalize scores before aggregation to robustly combine statistical and deep-learning signals.
- [ ] **Deep MIL Diversity**: Add attention orthogonality loss to prevent collapse to trivial sequences.

#### Phase 6 — Kaggle Submission Protocol (Daily until deadline)
Because we only get 5 submissions/day:
-   Submit only after passing Phase 0 checks.
-   Use a strict naming scheme: include git hash + timestamp + method tag in submission filename.
-   After each submission, update the ledger with Kaggle score and notes; do not repeat identical configs.

### 9.5 Definition of “Success”
We consider the approach on track if we see:
-   Strong per-dataset AUC on many datasets (ideally >0.80 on most),
-   High stability of Task 2 lists across folds/seeds,
-   Clear uplift from enrichment engine and from blending (vs. any single model).

---

## 10. Rules and Eligibility
-   Open Source (MIT) required for prizes.
-   Top 10 for paper authorship.
-   5 submissions per day.

---

## 11. References
-   [Competition Website](https://uio-bmi.github.io/adaptive_immune_profiling_challenge_2025/)
-   [Kaggle Page](https://www.kaggle.com/competitions/adaptive-immune-profiling-challenge-2025)

---

## 12. Verified Competition Details
-   **Deadline**: Dec 17, 2025 (07:59 AM CET). (~3.5 days remaining).
-   **Submission Limit**: 5 per day.
