#!/bin/bash
#SBATCH --job-name=smiles_train
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=0
#SBATCH --time=30-00:00:00
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

echo "Job started on $(hostname) at $(date)"

# ---------------------------------------------------------------------
# 1. Environment Setup
# ---------------------------------------------------------------------
source $HOME/miniconda/bin/activate battery_train_env
cd $SLURM_SUBMIT_DIR
mkdir -p logs
export PYTHONPATH=$PYTHONPATH:$(pwd)

# ---------------------------------------------------------------------
# 2. System Info & Debugging
# ---------------------------------------------------------------------
echo "Environment: $CONDA_DEFAULT_ENV"
echo "Python: $(which python)"
echo "GPUs Available: $CUDA_VISIBLE_DEVICES"
echo "SLURM Job ID: $SLURM_JOB_ID"

# ---------------------------------------------------------------------
# 3. Run Training
# ---------------------------------------------------------------------
python -u train/train_GATv2.py

echo "Job finished at $(date)"