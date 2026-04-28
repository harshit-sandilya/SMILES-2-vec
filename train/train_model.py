import os

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy

from .config import (
    BATCH_SIZE,
    EPOCHS,
    HIDDEN_DIM,
    LEARNING_RATE,
    NUM_HEADS,
    NUM_LAYERS,
    NUM_WORKERS,
    PROPERTY_LOSS_WEIGHT,
    VAL_CHECK_STEPS,
    WARMUP_STEPS,
)
from .data_module import MoleculeDataModule
from .lightning_model import GraphMoleculeLightningGATv2

# ---------------- Paths ----------------
BASE_DATA_DIR = "./data/optimized"
BASE_RESULTS_DIR = "./results"
BASE_MODELS_DIR = os.path.join(BASE_RESULTS_DIR, "models")
BASE_LOGS_DIR = os.path.join(BASE_RESULTS_DIR, "logs")

torch.set_float32_matmul_precision("high")
os.makedirs(BASE_MODELS_DIR, exist_ok=True)
os.makedirs(BASE_LOGS_DIR, exist_ok=True)


def main():

    pl.seed_everything(42)

    # ---------------- Data ----------------
    datamodule = MoleculeDataModule(
        data_dir=BASE_DATA_DIR,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )

    # ---------------- Model ----------------
    model = GraphMoleculeLightningGATv2(
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        lr=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        property_loss_weight=PROPERTY_LOSS_WEIGHT,
    )

    print("--- Training GATv2 Molecular Encoder ---")
    print("====================================")
    print(f"Hidden dim:            {HIDDEN_DIM}")
    print(f"Layers:                {NUM_LAYERS}")
    print(f"Heads:                 {NUM_HEADS}")
    print(f"Batch size:            {BATCH_SIZE}")
    print(f"LR:                    {LEARNING_RATE}")
    print(f"Property loss weight:  {PROPERTY_LOSS_WEIGHT}")
    print("====================================\n")

    # ---------------- Callbacks ----------------
    checkpoint_callback = ModelCheckpoint(
        dirpath=BASE_MODELS_DIR,
        filename="gatv2-{step:06d}-{val_loss:.4f}",
        monitor="val_loss",
        save_top_k=3,
        save_last=True,
        mode="min",
        every_n_train_steps=VAL_CHECK_STEPS,
    )

    early_stopping_callback = EarlyStopping(
        monitor="val_loss",
        patience=5,
        mode="min",
        verbose=True,
    )

    progress_bar = TQDMProgressBar(refresh_rate=100)

    # ---------------- Logger ----------------
    tensorboard_logger = TensorBoardLogger(
        save_dir=BASE_LOGS_DIR,
        name="gatv2_training",
    )

    # ---------------- Trainer ----------------
    trainer = pl.Trainer(
        strategy=DDPStrategy(find_unused_parameters=True),
        max_epochs=EPOCHS,
        accelerator="gpu",
        devices=-1,
        precision="bf16-mixed",
        gradient_clip_val=1.0,
        callbacks=[checkpoint_callback, early_stopping_callback, progress_bar],
        logger=tensorboard_logger,
        log_every_n_steps=50,
        val_check_interval=VAL_CHECK_STEPS,
    )

    print("Starting training...")
    trainer.fit(model, datamodule=datamodule)
    print("Training completed")

    # ---------------- Test ----------------
    print("Running test evaluation...")
    test_results = trainer.test(model, datamodule=datamodule)
    print(f"Test results: {test_results}")

    # ---------------- Save final model ----------------
    final_model_path = os.path.join(BASE_MODELS_DIR, "final_model_gatv2.ckpt")
    trainer.save_checkpoint(final_model_path)
    print(f"Final model saved to: {final_model_path}")


if __name__ == "__main__":
    main()
