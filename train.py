import os

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping

from config import *
from data_module import MoleculeDataModule
from lightning_model import GraphMoleculeLightning

model_dir = "checkpoints/"
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

if __name__ == "__main__":
    pl.seed_everything(42)
    datamodule = MoleculeDataModule(
        data_dir="optimized_graph_dataset",
        batch_size=64,
        num_workers=os.cpu_count() or 1,
    )
    model = GraphMoleculeLightning(
        hidden_dim=hidden_dim, num_layers=num_layers, num_heads=num_heads
    )
    checkpoint_callback = ModelCheckpoint(
        dirpath=model_dir,
        filename="model-{epoch:02d}",
        every_n_epochs=2,
        save_top_k=-1,
    )
    tensorboard_logger = TensorBoardLogger(
        save_dir="lightning_logs/", name="graph_molecule_model"
    )
    early_stopping_callback = EarlyStopping(
        monitor="val_loss", patience=3, verbose=True, mode="min"
    )
    trainer = pl.Trainer(
        max_epochs=-1,
        accelerator="cpu",
        callbacks=[checkpoint_callback, early_stopping_callback],
        logger=tensorboard_logger,
        log_every_n_steps=10,
    )
    print("Starting training on CPU...")
    trainer.fit(model, datamodule=datamodule)
    print("Training completed.")

    trainer.save_checkpoint("checkpoints/final_model.ckpt")
