#!/bin/bash
#SBATCH --job-name=smiles_processing
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --gres=gpu:0
#SBATCH --ntasks-per-node=224
#SBATCH --mem=0
#SBATCH --time=7-24:00:00
#SBATCH --output=process_%j.out
#SBATCH --error=process_%j.err

echo "--- Data Process Pipeline Started on $(hostname) ---"
date

# Load environment
source $HOME/miniconda/bin/activate battery_train_env
cd $SLURM_SUBMIT_DIR

# Stop on error
set -e

# --- CONFIGURATION FOR MPI STEPS ---
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1

python optimise_graph.py

echo "--- Data Process Pipeline Complete ---"
date