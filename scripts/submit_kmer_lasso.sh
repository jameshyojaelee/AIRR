#!/bin/bash
#SBATCH --job-name=airrml-kmer
#SBATCH --output=logs/airrml-kmer-%j.out
#SBATCH --error=logs/airrml-kmer-%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --partition=cpu

export PYTHONPATH=$PYTHONPATH:.
python3 scripts/run_experiment.py --config configs/kmer_lasso.json
