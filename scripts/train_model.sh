#!/bin/bash
#SBATCH --job-name=gatv2_pretrain
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=28
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

export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export RDKIT_LOGLEVEL=ERROR

python -u train/train_model.py

echo ""
echo "================================================================"
echo "  Pre-training complete  |  $(date)"
echo "  Checkpoints in: ./results/models/"
echo "  TensorBoard in: ./results/logs/"
echo "================================================================"
