import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from config import *
from train.data_module import MoleculeDataModule
from train.lightning_model_GCN import GraphMoleculeLightningGCN

BASE_DATA_DIR = "data"
BASE_RESULTS_DIR = "results"
BASE_MODELS_DIR = os.path.join(BASE_RESULTS_DIR, "models")
BASE_LOGS_DIR = os.path.join(BASE_RESULTS_DIR, "logs")

os.makedirs(BASE_MODELS_DIR, exist_ok=True)
os.makedirs(BASE_LOGS_DIR, exist_ok=True)

if __name__ == "__main__":
    pl.seed_everything(42)

    datamodule = MoleculeDataModule(
        data_dir=os.path.join(BASE_DATA_DIR, "optimized_graph_dataset"),
        batch_size=8,          # safe for RTX 3050
        num_workers=2,
    )

    model = GraphMoleculeLightningGCN(
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        lr=1e-4,
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=BASE_MODELS_DIR,
        filename="gcn-{epoch:02d}",
        every_n_epochs=2,
        save_top_k=-1,
    )

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=5,
        mode="min",
    )

    logger = TensorBoardLogger(
        save_dir=BASE_LOGS_DIR,
        name="graph_molecule_model_GCN",
    )

    trainer = pl.Trainer(
        max_epochs=30,
        accelerator="gpu",
        devices=1,
        callbacks=[checkpoint_callback, early_stopping],
        logger=logger,
        log_every_n_steps=10,
    )

    trainer.fit(model, datamodule=datamodule)
    trainer.save_checkpoint(os.path.join(BASE_MODELS_DIR, "final_model_GCN.ckpt"))


