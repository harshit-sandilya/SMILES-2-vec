#!/bin/bash
#SBATCH --job-name=smiles_download
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --gres=gpu:0
#SBATCH --ntasks-per-node=224
#SBATCH --mem=0
#SBATCH --time=10-00:00:00
#SBATCH --output=logs/download_%j.out
#SBATCH --error=logs/download_%j.err

echo "========================================================"
echo "  SMILES Download Pipeline"
echo "  Host    : $(hostname)"
echo "  Job ID  : $SLURM_JOB_ID"
echo "  Started : $(date)"
echo "========================================================"

# ── Stop on any error ─────────────────────────────────────────────────────
set -e
cd $SLURM_SUBMIT_DIR
mkdir -p logs

# ── Activate conda env ────────────────────────────────────────────────────
source "$HOME/miniconda/bin/activate" smiles

echo ""
echo "[env] Python  → $(which python) ($(python --version))"
echo "[env] mpirun  → $(which mpirun)"
echo "[env] OpenMPI → $(mpirun --version 2>&1 | head -1)"

# ── MPI / threading env vars ──────────────────────────────────────────────
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1

PYTHON=$(which python)
NPROCS=${SLURM_NTASKS:-224}
ZINC_PROCS=$(( NPROCS < 4 ? NPROCS : 4 ))

echo ""
echo "[config] Python → $PYTHON"
echo "[config] Ranks  → $NPROCS"
echo ""

# ── Helper: timed mpirun step ─────────────────────────────────────────────
run_step() {
    local label=$1
    local nprocs=$2
    local script=$3
    shift 3

    echo "--------------------------------------------------------"
    echo "  $label (ranks: $nprocs)"
    echo "  Start: $(date)"
    echo "--------------------------------------------------------"

    mpirun --bind-to none -np "$nprocs" "$PYTHON" "$script" "$@"

    echo "--------------------------------------------------------"
    echo "  $label"
    echo "  End: $(date)"
    echo "--------------------------------------------------------"
}

# ── Run downloads sequentially ────────────────────────────────────────────
run_step "[1/3] GDB13"   "$NPROCS" download/download_gdb13.py   --override
run_step "[2/3] MolPILE" "$NPROCS" download/download_molpile.py --override
run_step "[3/3] ZINC"    "$ZINC_PROCS" download/download_zinc.py    --override

# ── Summary ───────────────────────────────────────────────────────────────
echo "========================================================"
echo "  All downloads complete"
echo "  Finished : $(date)"
echo "========================================================"
