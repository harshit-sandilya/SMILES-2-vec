import os
import config
print(config.__file__)
print(dir(config))
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping

from config import *
from train.data_module import MoleculeDataModule
from train.lightning_model import GraphMoleculeLightning
import torch
torch.cuda.empty_cache()

BASE_DATA_DIR = "data"
BASE_RESULTS_DIR = "results"
BASE_MODELS_DIR = os.path.join(BASE_RESULTS_DIR, "models")
BASE_LOGS_DIR = os.path.join(BASE_RESULTS_DIR, "logs")

model_dir = BASE_MODELS_DIR

os.makedirs(model_dir, exist_ok=True)

if __name__ == "__main__":
    pl.seed_everything(42)
    datamodule = MoleculeDataModule(
        data_dir="data",
        batch_size=8,
        # batch_size=64,
        # num_workers=os.cpu_count() or 1,
        num_workers=2   ,
    )
    model = GraphMoleculeLightning(
        hidden_dim=config.HIDDEN_DIM, num_layers=config.NUM_LAYERS, num_heads=config.NUM_HEADS
    )
    checkpoint_callback = ModelCheckpoint(
        dirpath=model_dir,
        filename="model-{epoch:02d}",
        every_n_epochs=2,
        save_top_k=-1,
    )
    tensorboard_logger = TensorBoardLogger(
    save_dir=BASE_LOGS_DIR, name="graph_molecule_model"
    )
    
    early_stopping_callback = EarlyStopping(
        monitor="val_loss", patience=3, verbose=True, mode="min"
    )
    os.makedirs(BASE_LOGS_DIR, exist_ok=True)

    trainer = pl.Trainer(
        max_epochs=30,
        accelerator="gpu",
        devices=1,
        callbacks=[checkpoint_callback, early_stopping_callback],
        logger=tensorboard_logger,
        log_every_n_steps=10,
    )
    # print("Starting training on CPU...")
    trainer.fit(model, datamodule=datamodule)
    print("Training completed.")

    trainer.save_checkpoint(os.path.join(BASE_MODELS_DIR, "final_model.ckpt"))

