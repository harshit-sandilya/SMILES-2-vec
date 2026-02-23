import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

# Import SMART config for GATv2
from config_GATv2 import *
from train.data_module_GATv2 import MoleculeDataModule  # Use fixed GIN data module
from train.lightning_model_GATv2 import GraphMoleculeLightningGATv2


BASE_DATA_DIR = "data"
BASE_RESULTS_DIR = "results"
BASE_MODELS_DIR = os.path.join(BASE_RESULTS_DIR, "models")
BASE_LOGS_DIR = os.path.join(BASE_RESULTS_DIR, "logs")

os.makedirs(BASE_MODELS_DIR, exist_ok=True)
os.makedirs(BASE_LOGS_DIR, exist_ok=True)


if __name__ == "__main__":
    pl.seed_everything(42)

    # ---------------- Data ----------------
    # Use the FIXED data module (same as GIN) with tokenizer + masking
    datamodule = MoleculeDataModule(
        data_dir=BASE_DATA_DIR,  # Uses data/canonical_smiles.csv
        batch_size=BATCH_SIZE,
        num_workers=4,
        mask_ratio_atoms=0.15,
        mask_ratio_bonds=0.15,
    )

    # ---------------- Model (SMART GATv2) ----------------
    model = GraphMoleculeLightningGATv2(
        hidden_dim=HIDDEN_DIM,       # 512 from config_smart_gatv2.py
        num_layers=NUM_LAYERS,       # 6 from config_smart_gatv2.py
        num_heads=NUM_HEADS,         # 8 from config_smart_gatv2.py
        lr=LEARNING_RATE,
        use_contrastive=USE_CONTRASTIVE_LOSS,           # From config
        contrastive_weight=CONTRASTIVE_WEIGHT,          # From config
        contrastive_temperature=CONTRASTIVE_TEMPERATURE # From config
    )

    print(f"🚀 Training SMART GATv2 Model:")
    print(f"   Hidden dim: {HIDDEN_DIM}")
    print(f"   Layers: {NUM_LAYERS}")
    print(f"   Attention heads: {NUM_HEADS}")
    print(f"   Batch size: {BATCH_SIZE}")
    print(f"   Learning rate: {LEARNING_RATE}")
    print(f"   Contrastive loss: {USE_CONTRASTIVE_LOSS}")

    # ---------------- Callbacks ----------------
    checkpoint_callback = ModelCheckpoint(
        dirpath=BASE_MODELS_DIR,
        filename="gatv2_smart-{epoch:02d}",  # Different name from old model
        monitor="val_loss",
        every_n_epochs=2,
        save_top_k=3,
        mode="min",
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
        name="graph_molecule_model_GATv2_SMART",  # Different log directory
    )

    # ---------------- Trainer ----------------
    trainer = pl.Trainer(
        max_epochs=EPOCHS,  # 60 from config_smart_gatv2.py
        accelerator="gpu",
        devices=1,
        precision="bf16-mixed",
        callbacks=[checkpoint_callback, early_stopping_callback],
        logger=tensorboard_logger,
        log_every_n_steps=50,
    )

    print("🚀 Starting SMART GATv2 training on GPU...")
    trainer.fit(model, datamodule=datamodule)
    print("✅ Training completed.")

    # Evaluate on test set
    print("📊 Evaluating on test set...")
    test_results = trainer.test(model, datamodule=datamodule)
    print(f"Test results: {test_results}")

    # Save with different name
    trainer.save_checkpoint(
        os.path.join(BASE_MODELS_DIR, "final_model_GATv2_SMART.ckpt")  # Different name
    )
    
    print(f"\n✅ SMART GATv2 model saved to: {BASE_MODELS_DIR}/final_model_GATv2_SMART.ckpt")