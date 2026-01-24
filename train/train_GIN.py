# import sys
# from pathlib import Path

# PROJECT_ROOT = Path(__file__).resolve().parents[1]
# sys.path.insert(0, str(PROJECT_ROOT))

# import os
# import torch
# import pytorch_lightning as pl

# from config import *
# from train.lightning_model_GIN import GraphMoleculeLightningGIN

# # =========================
# # Paths
# # =========================
# BASE_RESULTS_DIR = "results"
# BASE_MODELS_DIR = os.path.join(BASE_RESULTS_DIR, "models")
# os.makedirs(BASE_MODELS_DIR, exist_ok=True)

# if __name__ == "__main__":
#     pl.seed_everything(42)

#     print("🧠 Initializing GIN encoder (no training)...")

#     model = GraphMoleculeLightningGIN(
#         hidden_dim=hidden_dim,
#         num_layers=num_layers,
#     )

#     ckpt_path = os.path.join(BASE_MODELS_DIR, "final_model_GIN.ckpt")

#     # ✅ Correct way for inference-only models
#     torch.save(model.state_dict(), ckpt_path)

#     print(f"✅ Saved GIN encoder weights to {ckpt_path}")

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
from train.data_module import MoleculeDataModule
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
        data_dir=os.path.join(BASE_DATA_DIR, "optimized_graph_dataset"),
        batch_size=8,          # safe for RTX 3050
        num_workers=2,
    )

    # ---------------- Model ----------------
    model = GraphMoleculeLightningGIN(
        hidden_dim=256,
        num_layers=5,
        lr=1e-4,
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
        max_epochs=50,
        accelerator="gpu",
        devices=1,
        callbacks=[checkpoint_callback, early_stopping_callback],
        logger=tensorboard_logger,
        log_every_n_steps=10,
    )

    print("🚀 Starting REAL GIN training on GPU...")
    trainer.fit(model, datamodule=datamodule)
    print("✅ Training completed.")

    trainer.save_checkpoint(
        os.path.join(BASE_MODELS_DIR, "final_model_GIN.ckpt")
    )




