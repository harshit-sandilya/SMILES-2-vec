import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from config import *
from train.data_module_GIN import MoleculeDataModule
from train.lightning_model_GIN import GraphMoleculeLightningGIN


BASE_DATA_DIR = "data"
BASE_RESULTS_DIR = "results"
BASE_MODELS_DIR = os.path.join(BASE_RESULTS_DIR, "models")
BASE_LOGS_DIR = os.path.join(BASE_RESULTS_DIR, "logs")

os.makedirs(BASE_MODELS_DIR, exist_ok=True)
os.makedirs(BASE_LOGS_DIR, exist_ok=True)


if __name__ == "__main__":
    pl.seed_everything(42)

    # ---------------- Data ----------------
    datamodule = MoleculeDataModule(
        data_dir="data",
        batch_size=16,
        num_workers=4,
        mask_ratio_atoms=0.15,
        mask_ratio_bonds=0.15,
    )

    # ---------------- Model ----------------
    model = GraphMoleculeLightningGIN(
        hidden_dim=HIDDEN_DIM,       # pulled from config (128)
        num_layers=NUM_LAYERS,       # FIX #9: pulled from config (5), was hardcoded
        lr=LEARNING_RATE,
    )

    # ---------------- Callbacks ----------------
    checkpoint_callback = ModelCheckpoint(
        dirpath=BASE_MODELS_DIR,
        filename="gin-{epoch:02d}",
        every_n_epochs=2,
        save_top_k=-1,
    )

    early_stopping_callback = EarlyStopping(
        monitor="val_loss",
        patience=5,
        verbose=True,
        mode="min",
    )

    # ---------------- Logger ----------------
    tensorboard_logger = TensorBoardLogger(
        save_dir=BASE_LOGS_DIR,
        name="graph_molecule_model_GIN",
    )

    # ---------------- Trainer ----------------
    trainer = pl.Trainer(
        max_epochs=30,
        accelerator="gpu",
        devices=1,
        precision="bf16-mixed",      # FIX #10: was "32" with a comment claiming speed gain; bf16 actually gives that
        callbacks=[checkpoint_callback, early_stopping_callback],
        logger=tensorboard_logger,
        log_every_n_steps=50,
    )

    print("🚀 Starting GIN training on GPU...")
    trainer.fit(model, datamodule=datamodule)
    print("✅ Training completed.")

    trainer.save_checkpoint(
        os.path.join(BASE_MODELS_DIR, "final_model_GIN.ckpt")
    )
