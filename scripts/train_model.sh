#!/bin/bash
#SBATCH --job-name=gatv2_pretrain
#SBATCH --partition=h100
#SBATCH --nodes=9
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=224
#SBATCH --mem=0
#SBATCH --time=30-00:00:00
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

echo "================================================================"
echo "  GATv2 Molecular Encoder — Pre-training"
echo "  Job ID    : $SLURM_JOB_ID"
echo "  Nodes     : $SLURM_NNODES"
echo "  Tasks     : $SLURM_NTASKS  ($SLURM_NTASKS_PER_NODE per node)"
echo "  GPUs/node : $SLURM_GPUS_ON_NODE"
echo "  Node list : $SLURM_NODELIST"
echo "  Started   : $(date)"
echo "================================================================"

source "$HOME/miniconda/bin/activate" smiles

echo ""
echo "[env] Python -> $(which python) ($(python --version 2>&1))"
echo "[env] PyTorch -> $(python -c 'import torch; print(torch.__version__)')"
echo "[env] CUDA   -> $(python -c 'import torch; print(torch.version.cuda)')"
echo ""

# ── Resume config ─────────────────────────────────────────────────────
# Set to path of last.ckpt to resume, leave empty for fresh start
RESUME_CKPT=""
# RESUME_CKPT="./results/models/last.ckpt"
export RESUME_CKPT
# ──────────────────────────────────────────────────────────────────────

export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export RDKIT_LOGLEVEL=ERROR
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1

MASTER_HOST=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
case "$MASTER_HOST" in
  innmi1srh1-p003) MASTER_ADDR=100.81.5.16 ;;
  innmi1srh1-p013) MASTER_ADDR=100.81.5.14 ;;
  innmi1srh1-p016) MASTER_ADDR=100.81.5.17 ;;
  innmi1srh1-p017) MASTER_ADDR=100.81.5.18 ;;
  innmi1srh1-p020) MASTER_ADDR=100.81.5.19 ;;
  innmi1srh1-p021) MASTER_ADDR=100.81.5.11 ;;
  innmi1srh1-p022) MASTER_ADDR=100.81.5.12 ;;
  innmi1srh1-p023) MASTER_ADDR=100.81.5.15 ;;
  innmi1srh1-p028) MASTER_ADDR=100.81.5.20 ;;
  *)
    MASTER_ADDR="$(getent hosts "$MASTER_HOST" | awk '{print $1}' | head -n 1)"
    if [[ "$MASTER_ADDR" == 127.* ]]; then
      echo "ERROR: MASTER_HOST=$MASTER_HOST resolved to loopback MASTER_ADDR=$MASTER_ADDR" >&2
      exit 2
    fi
    ;;
esac
export MASTER_ADDR
export MASTER_PORT=29500
export NCCL_SOCKET_IFNAME=bond0
export GLOO_SOCKET_IFNAME=bond0
export TP_SOCKET_IFNAME=bond0

echo "MASTER_ADDR=$MASTER_ADDR"
echo "MASTER_PORT=$MASTER_PORT"

export NCCL_IB_DISABLE=0
export NCCL_DEBUG=WARN
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export CUDA_DEVICE_MAX_CONNECTIONS=1

srun hostname

echo "--------------------------------------------------------"
echo "  Launching with srun  |  $(date)"
echo "--------------------------------------------------------"

srun --label bash -c '
set -euo pipefail
mkdir -p logs
exec > >(tee -a logs/python_${SLURM_JOB_ID}_node${SLURM_PROCID}.out) 2> >(tee -a logs/python_${SLURM_JOB_ID}_node${SLURM_PROCID}.err >&2)
echo "NODE_RANK=$SLURM_PROCID HOST=$(hostname) MASTER_ADDR='"$MASTER_ADDR"' MASTER_PORT='"$MASTER_PORT"'"

python -u -m torch.distributed.run \
    --nnodes='"$SLURM_NNODES"' \
    --nproc_per_node=8 \
    --node_rank=$SLURM_PROCID \
    --master_addr='"$MASTER_ADDR"' \
    --master_port='"$MASTER_PORT"' \
    --module train.train_model
'

echo ""
echo "================================================================"
echo "  Pre-training complete  |  $(date)"
echo "  Checkpoints in: ./results/models/"
echo "  TensorBoard in: ./results/logs/"
echo "================================================================"
