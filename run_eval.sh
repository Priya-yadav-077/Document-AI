#!/bin/bash

# Slurm script for running evaluation script for QASPER

#SBATCH --job-name=qasper_eval
#SBATCH --output=qasper_eval_%j.out
#SBATCH --error=qasper_eval_%j.err
#SBATCH --time=00:20:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=sari@cl.uni-heidelberg.de

# Load necessary modules (if required)
# module load python/3.11

# Activate conda/virtual environment (if you have one)
source ~/miniconda3/etc/profile.d/conda.sh
conda activate chroma

# Run the evaluation script
python qasper-modern-baselines/qasper_eval.py


