import os
import json
import pandas as pd
from pathlib import Path
from airrml import config, data
from airrml.submission import build_repertoire_predictions, build_sequence_importance, _load_model_artifacts

def assemble_single_dataset(train_root, test_root, model_name, model_output_root, dataset_name, top_k_sequences):
    model_dir = Path(model_output_root) / dataset_name
    print(f"[{dataset_name}] Loading artifacts from {model_dir}...")
    model, feature_info = _load_model_artifacts(model_dir, model_name)
    
    dataset_map = data.list_datasets(train_root, test_root)
    info = dataset_map.get(dataset_name)
    if not info:
        print(f"[{dataset_name}] Dataset info not found in map!")
        return []

    rows = []
    
    # Task 2: Sequence Importance
    print(f"[{dataset_name}] Calculating sequence importance (Fast Path)...")
    train_seq_df, _ = data.load_full_dataset(info["train_path"])
    # Drop ID to trigger Fast Path (Treat sequences as solo instances)
    train_seq_df_flat = train_seq_df.drop(columns=["ID"], errors="ignore")
    seq_rows = build_sequence_importance(model, train_seq_df_flat, dataset_name, top_k=top_k_sequences, feature_info=feature_info)
    rows.append(seq_rows)
    
    # Task 1: Repertoire Predictions
    print(f"[{dataset_name}] Building repertoire predictions...")
    for test_path in info["test_paths"]:
        test_seq_df, test_meta_df = data.load_test_dataset(test_path)
        preds = build_repertoire_predictions(model, feature_info, test_seq_df, test_meta_df, dataset_name=test_path.name)
        rows.append(preds)
        
    return rows

def main():
    train_root = "/gpfs/commons/home/jameslee/AIRR/train_datasets"
    test_root = "/gpfs/commons/home/jameslee/AIRR/test_datasets/test_datasets"
    model_name = "deep_mil"
    model_output_root = "outputs/deepmil_finetune_v1-12762703-gpu"
    top_k_sequences = 50000
    output_csv = Path(model_output_root) / "submission.csv"
    checkpoint_dir = Path(model_output_root) / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    dataset_names = [f"train_dataset_{i}" for i in range(1, 9)]
    all_rows = []

    for ds in dataset_names:
        ckpt = checkpoint_dir / f"{ds}_rows.joblib"
        if ckpt.exists():
            print(f"Found checkpoint for {ds}, loading...")
            rows = joblib.load(ckpt)
            all_rows.extend(rows)
            continue
        
        try:
            rows = assemble_single_dataset(train_root, test_root, model_name, model_output_root, ds, top_k_sequences)
            import joblib
            joblib.dump(rows, ckpt)
            all_rows.extend(rows)
        except Exception as e:
            print(f"FAILED on {ds}: {e}")
            import traceback
            traceback.print_exc()

    if all_rows:
        print(f"Concatenating {len(all_rows)} row blocks...")
        sub_df = pd.concat(all_rows, ignore_index=True)
        sub_df = sub_df[config.SUBMISSION_COLUMNS]
        sub_df.to_csv(output_csv, index=False)
        print(f"SUCCESS: Written to {output_csv}")
    else:
        print("ERROR: No rows generated.")

if __name__ == "__main__":
    main()
