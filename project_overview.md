# AIRR-ML-25: Adaptive Immune Profiling Challenge 2025

Project overview for participating in the AIRR‑ML‑25 Kaggle competition using this repository.

---

## 1. Challenge summary

AIRR‑ML‑25 is a machine learning challenge on adaptive immune receptor repertoires (AIRR). Participants build models that, for multiple datasets:

1. Predict the immune state of individuals or samples (for example disease vs healthy) from their adaptive immune receptor repertoires.
2. Identify receptor sequences that are most associated with that immune state.

The overarching goal is to accelerate development of machine learning methods for immunodiagnostics and therapeutics discovery. :contentReference[oaicite:0]{index=0}  

---

## 2. Timeline and platform

- **Platform**  
  - Hosted on Kaggle: `https://www.kaggle.com/competitions/adaptive-immune-profiling-challenge-2025` :contentReference[oaicite:1]{index=1}  

- **Key dates**  
  - Start: **05 November 2025, 08:00 CET**  
  - Final submission deadline: **17 December 2025, 07:59 CET** :contentReference[oaicite:2]{index=2}  

All deadlines use Central European Time (CET). Organizers reserve the right to adjust the timeline if needed. :contentReference[oaicite:3]{index=3}  

- **How to join**  
  - Create a Kaggle account.  
  - Open the competition page and accept the competition rules.  
  - Work in Kaggle notebooks or via the Kaggle API and submit the required predictions file. :contentReference[oaicite:4]{index=4}  

---

## 3. Prizes and publication

### 3.1 Monetary prizes

- **1st place**: 5,000 USD  
- **2nd place**: 3,000 USD  
- **3rd place**: 2,000 USD :contentReference[oaicite:5]{index=5}  

To be eligible for prize money, participants must release their competition code as open source. :contentReference[oaicite:6]{index=6}  

The monetary prizes are sponsored by **The Research Council of Norway**. :contentReference[oaicite:7]{index=7}  

### 3.2 Scientific manuscript authorship

- The **top 10 teams** on the final Kaggle leaderboard will be invited to contribute method descriptions, discussion, and code to a scientific article summarizing the competition outcome.  
- The article is already **accepted in principle at Nature Methods**. :contentReference[oaicite:8]{index=8}  

---

## 4. Formal problem definition

You are given multiple datasets of T cell receptor (TCR) repertoires. Each repertoire belongs to an individual or sample and contains many receptor sequences. For each dataset, you must solve two tasks.

### 4.1 Task 1: Immune state prediction

- **Input**  
  - For each `repertoire_id` in the test data, you see the corresponding receptor sequences (for example a `.tsv` file per repertoire).  

- **Output**  
  - For every `repertoire_id` across all test datasets, you must submit a **probability** that the repertoire is **label positive** (for example disease present). :contentReference[oaicite:9]{index=9}  

This is evaluated with area under the ROC curve (AUC) per dataset (see section 6).

### 4.2 Task 2: Label associated sequence identification

- **Input**  
  - Training datasets with known labels.  

- **Output**  
  - For each **training dataset**, you must submit a **ranked list** of up to **50,000 unique rows**, each defined by: :contentReference[oaicite:10]{index=10}  
    - `junction_aa`  
    - `v_call`  
    - `j_call`  

  - The rows must be sorted from most important to less important according to some importance score derived from your model or post hoc analysis.  
  - The organizers may only use the top part of this list when computing the sequence level metric. :contentReference[oaicite:11]{index=11}  

This is evaluated using Jaccard similarity between your submitted set of sequences and a hidden reference set per dataset (see section 6). :contentReference[oaicite:12]{index=12}  

---

## 5. Datasets and data sources

### 5.1 Number of datasets

- **Training**: 8 datasets  
- **Test**: 10 datasets :contentReference[oaicite:13]{index=13}  

Each dataset represents a distinct cohort, experimental setting, or synthetic setup. Evaluation is performed per dataset and then aggregated.

### 5.2 Biological data sources

The challenge combines experimental and synthetic TCR data. As described in the official documentation: :contentReference[oaicite:14]{index=14}  

- **Adaptive Biotechnologies**  
  - Provides approximately **500 unpublished TCRβ repertoires** from donors with known **HSV‑2 infection status**.  

- **Parse Biosciences**  
  - Provides **antigen specific TCR sequences** from an experiment that profiled around **1 million antigen reactive human T cells**.  
  - These data are used to construct synthetic datasets for the competition.  

This design allows testing methods on both real and controlled synthetic settings while preserving realistic repertoire properties.

### 5.3 On disk layout in the Kaggle runtime

The organizers' example notebooks and code template assume the following paths in the Kaggle environment:

- **Root competition data directory**  
  - `/kaggle/input/adaptive-immune-profiling-challenge-2025/`

- **Training datasets**  
  - `/kaggle/input/adaptive-immune-profiling-challenge-2025/train_datasets/train_datasets/`  
  - Contains subdirectories such as `train_dataset_1`, `train_dataset_2`, etc.  

- **Test datasets**  
  - `/kaggle/input/adaptive-immune-profiling-challenge-2025/test_datasets/test_datasets/`  
  - Contains subdirectories such as `test_dataset_1_1`, `test_dataset_1_2`, etc, linked to the training datasets by ID via the code template.

- **Default output directory for your runs**  
  - `/kaggle/working/results/`  

These exact paths are hard coded in the organizer notebooks and are intended as the standard layout for submissions built on the template.

### 5.4 Per dataset contents and metadata

The official code template utilities are written to handle two cases:

1. **With metadata**  
   - A file `metadata.csv` in each dataset directory describes the repertoires and, for training sets, their labels. The template expects columns including:
     - `repertoire_id`
     - `filename` (name of the `.tsv` file with sequences)
     - `label_positive` (binary outcome, for example 0 or 1, in training data).

2. **Without metadata**  
   - If `metadata.csv` is not present, the utilities fall back to scanning all `*.tsv` files in the directory and derive repertoire IDs from filenames.

Sequence level `.tsv` files contain the receptors for each repertoire. The downstream evaluation and submission format require that you track at least the following columns:

- `junction_aa`  
- `v_call`  
- `j_call`  

The template’s utilities (`load_data_generator`, `load_full_dataset`, `load_and_encode_kmers`) are designed to read these structures and assemble both sequence level and repertoire level views of the data.

---

## 6. Evaluation and leaderboard metric

### 6.1 Per dataset metrics

For each dataset, two metrics are computed. :contentReference[oaicite:15]{index=15}  

1. **Immune state prediction (Task 1)**  
   - From your submission, the organizer extracts the probabilities provided for each `repertoire_id` in the test sets.  
   - Performance is measured with **area under the ROC curve (AUC)**.

2. **Label associated sequence identification (Task 2)**  
   - From the ranked list of up to 50,000 sequences you submit per training dataset, the organizer computes a **Jaccard similarity** between your set of sequences and their hidden set of label associated sequences.  
   - Only the top part of your ranked list may be used for evaluation.

Both AUC and Jaccard are computed separately for each dataset. :contentReference[oaicite:16]{index=16}  

### 6.2 Overall competition score and ranking

- A **weighted average** over datasets and over the two metrics (AUC and Jaccard) is used to define a single competition score per submission. :contentReference[oaicite:17]{index=17}  
- Kaggle’s leaderboard is based on this aggregate score. A subset of test data may be held out for the final ranking, following standard Kaggle practice.

---

## 7. Expected submission format

The organizers provide a helper that concatenates per dataset outputs into a single CSV file ready for upload.

The standard pattern is:

1. For each training dataset:
   - Save a TSV file of sequence level outputs, named like  
     `train_dataset_X_important_sequences.tsv`  
     with columns  
     `["ID", "dataset", "label_positive_probability", "junction_aa", "v_call", "j_call"]`.  

2. For each associated test dataset:
   - Save a TSV file of repertoire level predictions, named like  
     `train_dataset_X_test_predictions.tsv`  
     with the same six columns.

3. Run the provided helper `concatenate_output_files(results_dir)` to:  
   - Read all `*_test_predictions.tsv` and `*_important_sequences.tsv` files from `results_dir`.  
   - Concatenate them into a single `submissions.csv` file with schema:  
     `["ID", "dataset", "label_positive_probability", "junction_aa", "v_call", "j_call"]`.  

This `submissions.csv` is what you upload to Kaggle.

The exact column names and their meaning:

- `ID`  
  - Repertoire ID for test predictions.  
  - Synthetic identifier for sequence level entries (for example `train_dataset_1_seq_top_00001`).  

- `dataset`  
  - Name of the dataset directory (for example `train_dataset_1`, `test_dataset_1_1`).  

- `label_positive_probability`  
  - For test predictions: the model’s estimated probability that the repertoire is label positive.  
  - For sequence level entries: typically set to a placeholder value (for example `-999.0`) to satisfy the unified schema.  

- `junction_aa`, `v_call`, `j_call`  
  - TCR sequence and gene segment identifiers, required for the sequence level evaluation.

---

## 8. Official code template and baseline

### 8.1 Template utilities

The organizers supply a code template that defines:

- **Data utilities**  
  - `load_data_generator(data_dir, metadata_filename="metadata.csv")`  
  - `load_full_dataset(data_dir)`  
  - `load_and_encode_kmers(data_dir, k=3)`  

- **Helper functions**  
  - `save_tsv(df, path)`  
  - `validate_dirs_and_files(train_dir, test_dirs, out_dir)`  
  - `get_dataset_pairs(train_dir, test_dir)`  
  - `concatenate_output_files(out_dir)`  

These functions handle streaming repertoires from disk, simple k‑mer encoding, and assembling the final submission file.

### 8.2 Model interface

The core abstraction is the `ImmuneStatePredictor` class:

- `__init__(self, n_jobs=1, device="cpu", **kwargs)`  
  - Stores parallelism and device preferences.  

- `fit(self, train_dir_path)`  
  - Trains a model on a single training dataset (for example `train_dataset_1`).  
  - Is responsible for setting an attribute `self.important_sequences_`, containing a DataFrame of label associated sequences for that dataset with the required submission columns.  

- `predict_proba(self, test_dir_path)`  
  - Returns a DataFrame of repertoire level predictions with the schema described in section 7.  

- `identify_associated_sequences(...)`  
  - Encapsulates the logic for scoring sequences and selecting the top label associated ones.

### 8.3 Execution workflow

The template also provides a `main` function that:

1. Validates input and output directories.  
2. Instantiates `ImmuneStatePredictor`.  
3. Calls `fit` on the specified training dataset.  
4. Calls `predict_proba` for each associated test dataset.  
5. Writes `*_test_predictions.tsv` and `*_important_sequences.tsv` to `results_dir`.  

A convenience function `get_dataset_pairs(train_datasets_dir, test_datasets_dir)` returns a list of  
`(train_dataset_path, [associated_test_dataset_paths])`, allowing you to loop over all datasets.

At the end of a full run over all pairs, `concatenate_output_files(results_dir)` is used to produce the combined `submissions.csv`.

### 8.4 Example baseline approach

The organizer baseline notebooks demonstrate one simple strategy:

- Encode repertoire level features as counts of **3‑mer subsequences** from the `junction_aa` field.  
- Train an **L1 regularized logistic regression** on these features to predict `label_positive`.  
- Derive **sequence importance scores** by matching k‑mer presence in each sequence against the learned coefficients, and summing contributions to obtain an importance score per sequence.  

This baseline satisfies the required outputs for both tasks and provides a reference point for model development.

---

## 9. Code Requirements

To win the prize money, a prerequisite is that the code has to be made open-source. In addition, the top 10 submissions/teams will be invited to become co-authors in a scientific paper.

To enable further analyses and re-use of the models by the community, participants are strongly encouraged to adhere to the code template provided in this repository.

It is also important to provide the exact requirements/dependencies to be able to containerize and run the code. A `requirements.txt` file is included in this repository for this purpose. If you modify the code or add dependencies, please update `requirements.txt` with your dependencies and exact versions.

---

## 10. Rules and eligibility (high level)

This section captures only what is explicitly stated in the public challenge documentation. For full legal terms, refer to the Kaggle competition rules page for AIRR‑ML‑25.

- **Participation**  
  - The challenge is **open to everyone** via the Kaggle platform, subject to Kaggle’s own account and rules acceptance process. :contentReference[oaicite:18]{index=18}  

- **Prize eligibility**  
  - To receive monetary prizes, teams must release their solution code as open source. :contentReference[oaicite:19]{index=19}  

- **Publication**  
  - The top 10 teams on the final leaderboard will be invited to join a scientific manuscript summarizing the competition, to be published in Nature Methods. :contentReference[oaicite:20]{index=20}  

- **Data providers and acknowledgements**  
  - Adaptive Biotechnologies and Parse Biosciences provide the key experimental data used to construct the training and test datasets, as described in section 5. :contentReference[oaicite:21]{index=21}  

Any additional constraints such as daily submission limits, team size, or external data policies must be taken from the Kaggle rules page directly, as they are not duplicated here.

---

## 11. References

- AIRR‑ML‑25 organizers, official challenge site:  
  `https://uio-bmi.github.io/adaptive_immune_profiling_challenge_2025/` :contentReference[oaicite:22]{index=22}  

- Kaggle competition page (primary host):  
  `https://www.kaggle.com/competitions/adaptive-immune-profiling-challenge-2025`  

These two sources together define the official competition description, evaluation criteria, and high level rules.

---

## 12. Competition details to verify (not yet captured here)

The Kaggle competition pages load detailed rules dynamically for signed-in users. The items below are not documented in this repository and should be confirmed directly on the competition site before finalizing submissions:

- Exact **weighting** used to combine AUC and Jaccard across datasets in the overall score.
- **Submission limits**: daily and total caps, and whether limits differ by phase.
- **Leaderboard protocol**: public/private split of test data and any multi-phase timeline.
- **External data policy**: whether outside data are permitted, disclosure requirements, and restrictions.
- **Code vs. prediction submission rules**: runtime limits, internet access, hardware constraints (CPU/GPU), and required artifacts for code submissions (if applicable).
- **Team rules**: team size limits, merge deadlines, and late-join restrictions.
- **Organizer resources**: any additional starter notebooks or baselines beyond the 3-mer logistic regression example.

Please update this document once those specifics are confirmed so the repository reflects the authoritative competition requirements.
