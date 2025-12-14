#!/usr/bin/env python3
"""
Monitor progress of AIRR-ML-25 training jobs by parsing Slurm logs.
"""
import re
import sys
from pathlib import Path
from typing import Dict, Any

def parse_log(path: Path) -> Dict[str, Any]:
    content = path.read_text(errors="replace")
    stats = {"name": path.name, "status": "Unknown", "epoch": None, "loss": None, "auc": None}
    
    # Check for completion
    if "Submission written to" in content:
        stats["status"] = "DONE"
    elif "Traceback (most recent call last)" in content or "Killed" in content:
        stats["status"] = "FAILED"
    else:
        stats["status"] = "RUNNING"
        
    # Extract Epoch/Loss (Deep MIL / Pretrain)
    # [Epoch 5] Val Loss: 0.6543
    # [Epoch 3/10] Loss: 1.2345
    epoch_matches = list(re.finditer(r"\[Epoch (\d+)[/\]]?(\d+)?\] (?:Val )?Loss: ([\d\.]+)", content))
    if epoch_matches:
        last = epoch_matches[-1]
        stats["epoch"] = last.group(1)
        stats["loss"] = last.group(3)
        
    # Extract Validation AUC (GBM / Deep MIL Summary)
    # "  train_dataset_1: 0.7834"
    auc_matches = list(re.finditer(r"^\s+(train_dataset_\d+): ([\d\.]+)", content, re.MULTILINE))
    if auc_matches:
        stats["aucs"] = {m.group(1): float(m.group(2)) for m in auc_matches}
        if stats["status"] == "RUNNING":
             # If running but we have AUCs, it might be partial
             stats["last_ds"] = auc_matches[-1].group(1)
             stats["last_auc"] = auc_matches[-1].group(2)
             
    # Extract Pretraining info
    if "Loaded" in content and "sequences for pretraining" in content:
         stats["type"] = "Pretrain"
         
    return stats

def main():
    log_dir = Path("logs")
    logs = sorted(log_dir.glob("airrml-*.out"), key=lambda p: p.stat().st_mtime, reverse=True)
    # Show top 10 most recent
    logs = logs[:10]
    
    print(f"{'Job Name':<35} | {'Status':<10} | {'Progress':<30}")
    print("-" * 80)
    
    for log in logs:
        info = parse_log(log)
        prog = ""
        if info.get("epoch"):
            prog = f"Epoch {info['epoch']}, Loss {info['loss']}"
        elif info.get("aucs"):
            done_cnt = len(info['aucs'])
            last = info.get('last_auc', '?')
            prog = f"{done_cnt} datasets done (Last AUC: {last})"
        elif info.get("type") == "Pretrain":
            prog = "Loading/Processing..."
            
        print(f"{info['name']:<35} | {info['status']:<10} | {prog:<30}")

if __name__ == "__main__":
    main()
