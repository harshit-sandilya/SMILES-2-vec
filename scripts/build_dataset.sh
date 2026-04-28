#!/bin/bash
#SBATCH --job-name=smiles_process
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --gres=gpu:0
#SBATCH --ntasks-per-node=1          # one process — Spark owns all threads
#SBATCH --cpus-per-task=224          # Spark local[N] — must match SPARK_CORES
#SBATCH --mem=0                      # claim all node memory
#SBATCH --time=10-00:00:00
#SBATCH --output=logs/process_%j.out
#SBATCH --error=logs/process_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

# ── Banner ────────────────────────────────────────────────────────────────
echo "========================================================"
echo "  SMILES Processing Pipeline (PySpark)"
echo "  Host    : $(hostname)"
echo "  Job ID  : $SLURM_JOB_ID"
echo "  Started : $(date)"
echo "========================================================"

# ── Conda env ─────────────────────────────────────────────────────────────
source "$HOME/miniconda/bin/activate" smiles

echo ""
echo "[env] Python  → $(which python) ($(python --version 2>&1))"

# ── Java (required by Spark) ──────────────────────────────────────────────
# PySpark ships its own Spark but needs a JVM on PATH.
# Try: conda env first, then module system, then fail with helpful message.
if ! command -v java &>/dev/null; then
    # conda-installed openjdk puts java here:
    CONDA_JAVA="$CONDA_PREFIX/bin/java"
    if [ -f "$CONDA_JAVA" ]; then
        export PATH="$(dirname "$CONDA_JAVA"):$PATH"
    else
        # fall back to cluster module system
        module load java 2>/dev/null   \
        || module load jdk  2>/dev/null \
        || module load openjdk 2>/dev/null \
        || {
            echo ""
            echo "ERROR: Java not found."
            echo "  Fix: conda install -n smiles -c conda-forge openjdk"
            exit 1
        }
    fi
fi

# Derive JAVA_HOME from the resolved java binary
export JAVA_HOME
JAVA_HOME="$(dirname "$(dirname "$(readlink -f "$(command -v java)")")")"
echo "[env] Java    → $(java -version 2>&1 | head -1)"
echo "[env] JAVA_HOME → $JAVA_HOME"

# ── Spark scratch directory ───────────────────────────────────────────────
# Use a fast local disk if available ($TMPDIR is set by SLURM on many HPC
# systems to a node-local SSD). Fall back to /tmp.
SPARK_TMP="${TMPDIR:-/tmp}/spark_${SLURM_JOB_ID}"
mkdir -p "$SPARK_TMP"
echo "[env] Spark tmp → $SPARK_TMP"

# ── Spark driver memory  ──────────────────────────────────────────────────
# Allocate 82 % of total physical RAM to the JVM driver.
# The remaining 18 % is headroom for:
#   - OS kernel + file system buffers
#   - RDKit native (C++) heap used by pandas_udf workers
#   - Python overhead per Arrow batch worker
TOTAL_MEM_KB=$(grep MemTotal /proc/meminfo | awk '{print $2}')
TOTAL_MEM_GB=$(( TOTAL_MEM_KB / 1024 / 1024 ))
DRIVER_MEM_GB=$(( TOTAL_MEM_GB * 82 / 100 ))
DRIVER_MEM="${DRIVER_MEM_GB}g"
echo "[env] Node RAM  → ${TOTAL_MEM_GB} GB"
echo "[env] Driver mem→ ${DRIVER_MEM}"

# ── CPU cores ─────────────────────────────────────────────────────────────
# SLURM_CPUS_PER_TASK is set by --cpus-per-task above.
# Use it so the script auto-scales if the SBATCH value changes.
CORES="${SLURM_CPUS_PER_TASK:-224}"
echo "[env] Cores     → $CORES"
echo ""

# ── Threading: prevent numpy / OpenBLAS from spawning extra threads ───────
# Each Spark Python worker already runs in its own OS thread;
# sub-threading inside workers wastes CPU and causes contention.
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# ── Pass extra CLI arguments straight through (e.g. --max-atoms 96) ───────
EXTRA_ARGS=("$@")

# ── Run ───────────────────────────────────────────────────────────────────
echo "--------------------------------------------------------"
echo "  process_raw_data.py"
echo "  Start : $(date)"
echo "--------------------------------------------------------"

python preprocess/build_dataset.py \
    --cores         "$CORES"      \
    --driver-memory "$DRIVER_MEM" \
    --local-dir     "$SPARK_TMP"  \
    "${EXTRA_ARGS[@]}"

echo "--------------------------------------------------------"
echo "  process_raw_data.py"
echo "  End   : $(date)"
echo "--------------------------------------------------------"

# ── Cleanup temp ──────────────────────────────────────────────────────────
rm -rf "$SPARK_TMP"

echo ""
echo "========================================================"
echo "  Processing complete"
echo "  Finished : $(date)"
echo "========================================================"
