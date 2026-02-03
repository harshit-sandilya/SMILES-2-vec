import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from config import *                                      # imports BATCH_SIZE, EPOCHS, HIDDEN_DIM, NUM_LAYERS, LEARNING_RATE
from train.data_module_GCN import MoleculeDataModuleGCN   # FIX: imports from data_module_GCN
from train.lightning_model_GCN import GraphMoleculeLightningGCN

BASE_DATA_DIR = "data"
BASE_RESULTS_DIR = "results"
BASE_MODELS_DIR = os.path.join(BASE_RESULTS_DIR, "models")
BASE_LOGS_DIR = os.path.join(BASE_RESULTS_DIR, "logs")

os.makedirs(BASE_MODELS_DIR, exist_ok=True)
os.makedirs(BASE_LOGS_DIR, exist_ok=True)

if __name__ == "__main__":
    pl.seed_everything(42)

    datamodule = MoleculeDataModuleGCN(
        data_csv=os.path.join(BASE_DATA_DIR, "canonical_smiles_subset_100k.csv"),  # FIX: data_module_GCN expects data_csv, not data_dir
        batch_size=BATCH_SIZE,        # FIX BUG 6: was hardcoded to 8, now uses config (64)
        num_workers=2,
    )

    model = GraphMoleculeLightningGCN(
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        lr=LEARNING_RATE,
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=BASE_MODELS_DIR,
        filename="gcn-{epoch:02d}",
        every_n_epochs=2,
        save_top_k=-1,
    )

    early_stopping = EarlyStopping(
        monitor="val_loss",           # Now monitors combined val_loss (recon + distinction)
        patience=5,
        mode="min",
    )

    logger = TensorBoardLogger(
        save_dir=BASE_LOGS_DIR,
        name="graph_molecule_model_GCN",
    )

    trainer = pl.Trainer(
        max_epochs=EPOCHS,            # FIX BUG 6: was hardcoded to 20, now uses config (30)
        accelerator="gpu",
        devices=1,
        callbacks=[checkpoint_callback, early_stopping],
        logger=logger,
        log_every_n_steps=10,
    )

    trainer.fit(model, datamodule=datamodule)
    trainer.save_checkpoint(os.path.join(BASE_MODELS_DIR, "final_model_GCN.ckpt"))
