#!/bin/bash
#SBATCH --job-name=smiles_optimise
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --gres=gpu:0
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=224
#SBATCH --mem=0
#SBATCH --time=10-00:00:00
#SBATCH --output=logs/optimise_%j.out
#SBATCH --error=logs/optimise_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

echo "========================================================"
echo "  SMILES Optimise Pipeline (litdata)"
echo "  Host    : $(hostname)"
echo "  Job ID  : $SLURM_JOB_ID"
echo "  CPUs    : ${SLURM_CPUS_PER_TASK}"
echo "  Started : $(date)"
echo "========================================================"

source "$HOME/miniconda/bin/activate" smiles

echo ""
echo "[env] Python -> $(which python) ($(python --version 2>&1))"
echo ""

# litdata workers communicate via shared memory files.
# If $TMPDIR is a fast local NVMe scratch (common on HPC), use it.
LITDATA_TMP="${TMPDIR:-/dev/shm}/litdata_${SLURM_JOB_ID}"
mkdir -p "$LITDATA_TMP"
export LITDATA_CACHE_DIR="$LITDATA_TMP"
echo "[env] litdata tmp -> $LITDATA_TMP"

# Each litdata worker is already a dedicated OS process.
# Allowing NumPy/OpenBLAS to spawn thread pools inside each worker
# causes 128 workers x 8 threads = 1024 threads on 224 cores.
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo ""
echo "--------------------------------------------------------"
echo "  optimise_dataset.py  |  Start: $(date)"
echo "--------------------------------------------------------"

python preprocess/optimise_dataset.py

echo "--------------------------------------------------------"
echo "  optimise_dataset.py  |  End:   $(date)"
echo "--------------------------------------------------------"

rm -rf "$LITDATA_TMP"

echo ""
echo "========================================================"
echo "  Optimise complete  |  $(date)"
echo "========================================================"
