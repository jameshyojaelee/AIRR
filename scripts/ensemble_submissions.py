"""
Ensemble multiple submission files by averaging probabilities and aggregating sequence ranks.

Usage:
    python3 scripts/ensemble_submissions.py \
        --submissions sub1.csv sub2.csv ... \
        --weights 1.0 0.5 ... \
        --output ensemble_submission.csv
"""
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List

def normalize_ranks(df: pd.DataFrame, top_k: int) -> pd.DataFrame:
    """
    Convert importance order to a normalized score (1.0 = top, 0.0 = bottom).
    Output has 'junction_aa', 'v_call', 'j_call', 'score'.
    """
    # Identify sequence rows (placeholder probability)
    # Using config constants would be better, but hardcoding for script independence
    seq_rows = df[df["label_positive_probability"].between(0.49, 0.51, inclusive="both")].copy()
    
    # We assume the input rows are ALREADY sorted by importance or correspond to the top-K
    # But actually, in the submission file, there is no explicit score column.
    # We must assume the ORDER in the file reflects the rank, or we can't ensemble.
    # The `outputs/` usually contain model objects with scores, but the CSV strips them.
    # WAIT: `assemble_submission` in `submission.py` outputs rows. It does NOT guarantee order in the final CSV 
    # because `pd.concat` is used.
    # HOWEVER, `format_sequence_rows` assigns IDs `_seq_top_0`, `_seq_top_1`...
    # We can parse the ID to recover the rank!
    
    seq_rows["rank"] = seq_rows["ID"].str.extract(r"_seq_top_(\d+)").astype(float)
    max_rank = seq_rows.groupby("dataset")["rank"].transform("max")
    seq_rows["score"] = 1.0 - (seq_rows["rank"] / (max_rank + 1))
    
    return seq_rows[["dataset", "junction_aa", "v_call", "j_call", "score"]]

def main():
    parser = argparse.ArgumentParser(description="Ensemble submissions")
    parser.add_argument("--submissions", nargs="+", required=True, help="List of submission CSV paths")
    parser.add_argument("--weights", nargs="+", type=float, help="Weights for each submission")
    parser.add_argument("--output", required=True, help="Output path")
    args = parser.parse_args()

    subs = [pd.read_csv(p) for p in args.submissions]
    weights = args.weights if args.weights else [1.0] * len(subs)
    if len(weights) != len(subs):
        raise ValueError("Number of weights must match number of submissions")

    # Normalize weights
    weights = np.array(weights) / sum(weights)

    print(f"Ensembling {len(subs)} submissions with weights {weights}...")

    # --- Task 1: Repertoire Probabilities ---
    print("Processing Task 1 (Repertoires)...")
    # Filter for repertoire rows (probability not 0.5 placeholder, or check ID)
    # Better: check for "seq" in ID
    rep_dfs = []
    for df, w in zip(subs, weights):
        # Repertoire rows have IDs that do NOT contain "_seq_top_"
        mask = ~df["ID"].str.contains("_seq_top_")
        d = df[mask].set_index(["ID", "dataset"])[["label_positive_probability"]]
        rep_dfs.append(d * w)
    
    # Sum weighted probabilities
    ens_rep = sum(rep_dfs)
    ens_rep = ens_rep.reset_index()
    # Fill required columns
    ens_rep["junction_aa"] = "X" * 15 # Placeholder
    ens_rep["v_call"] = "X"
    ens_rep["j_call"] = "X"
    
    # --- Task 2: Sequence Aggregation ---
    print("Processing Task 2 (Sequences)...")
    seq_scores = []
    for df, w in zip(subs, weights):
        # Extract rank-based scores
        mask = df["ID"].str.contains("_seq_top_")
        rows = df[mask].copy()
        # Parse rank from ID `..._seq_top_{rank}`
        rows["rank"] = rows["ID"].str.split("_seq_top_").str[-1].astype(int)
        
        # Normalize score: 1 / (rank + 1) or linear? 
        # Linear degradation is safer for robust lists.
        # Max rank per dataset
        max_ranks = rows.groupby("dataset")["rank"].max()
        rows = rows.merge(max_ranks.rename("max_rank"), on="dataset")
        rows["score"] = (1.0 - (rows["rank"] / (rows["max_rank"] + 1))) * w
        
        seq_scores.append(rows[["dataset", "junction_aa", "v_call", "j_call", "score"]])

    all_seqs = pd.concat(seq_scores)
    # Sum scores per sequence per dataset
    agg_seqs = all_seqs.groupby(["dataset", "junction_aa", "v_call", "j_call"])["score"].sum().reset_index()
    
    # Select top-K per dataset
    final_seq_rows = []
    # We need to know the target Top-K. Usually 50k for this challenge.
    TOP_K = 50000 
    
    for dataset, group in agg_seqs.groupby("dataset"):
        # Sort by score descending
        top = group.sort_values("score", ascending=False).head(TOP_K)
        
        # Format rows
        top["dataset"] = dataset
        top["label_positive_probability"] = 0.5
        # Generate IDs
        top["ID"] = [f"{dataset}_seq_top_{i}" for i in range(len(top))]
        
        final_seq_rows.append(top[["ID", "dataset", "label_positive_probability", "junction_aa", "v_call", "j_call"]])

    ens_seq = pd.concat(final_seq_rows) if final_seq_rows else pd.DataFrame()

    # --- Merge and Save ---
    final_df = pd.concat([ens_rep, ens_seq], ignore_index=True)
    # Ensure column order
    cols = ["ID", "dataset", "label_positive_probability", "junction_aa", "v_call", "j_call"]
    final_df = final_df[cols]
    
    print(f"Writing {len(final_df)} rows to {args.output}")
    final_df.to_csv(args.output, index=False)

if __name__ == "__main__":
    main()
