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

### 9.1 Status (as of Dec 13, 2025)
-   **Implemented**:
    -   Full pipeline for loading data, training, and generating submissions.
    -   `DeepMILModel` with PyTorch.
    -   `KmerLogReg` baseline.
    -   HPO via Optuna.
-   **Verified**:
    -   Data loading and encoding.
    -   Baseline model runs.

### 9.2 Tasks & Roadmap (Dec 13 - Dec 17)

#### Phase 1: Validation & Tuning (Immediate)
-   [ ] **Verify Deep MIL Performance**: Run `scripts/run_loco.py` with `configs/deepmil_transformer.json` to get a reliable CV AUC estimate.
-   [ ] **Hyperparameter Tuning**: Run `tune_optuna.py` for Deep MIL to optimize model dimension, heads, and dropout.
-   [ ] **Sequence Ranking Check**: Verify that `get_sequence_importance` in `DeepMILModel` produces reasonable outputs (Jaccard proxy).

#### Phase 2: Scaling & Ensembling
-   [ ] **Full Training**: Execute `train_all_datasets.py` on GPU nodes for final model candidates.
-   [ ] **Ensemble Strategy**: Combine predictions from `kmer_logreg` (linear) and `deep_mil` (non-linear).
    -   *Plan*: Average probabilities for Task 1. Merge sequence lists for Task 2 (e.g., take top 25k from each).

#### Phase 3: Submission
-   [ ] **Generate Submission**: Use `scripts/submit_assemble_deepmil_outputs.sh` or `generate_submission` in `pipeline.py`.
-   [ ] **Upload to Kaggle**.

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
