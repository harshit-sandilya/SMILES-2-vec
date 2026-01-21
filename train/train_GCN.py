import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import os
import torch
import pytorch_lightning as pl

from config import *
from train.lightning_model_GCN import GraphMoleculeLightningGCN

BASE_RESULTS_DIR = "results"
BASE_MODELS_DIR = os.path.join(BASE_RESULTS_DIR, "models")
os.makedirs(BASE_MODELS_DIR, exist_ok=True)

if __name__ == "__main__":
    pl.seed_everything(42)

    print("🧠 Initializing GCN encoder (no training)...")

    model = GraphMoleculeLightningGCN(
        hidden_dim=hidden_dim,
        num_layers=num_layers,
    )

    ckpt_path = os.path.join(BASE_MODELS_DIR, "final_model_GCN.ckpt")

    torch.save(model.state_dict(), ckpt_path)

    print(f"✅ Saved GCN encoder weights to {ckpt_path}")
