import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch_geometric.loader import DataLoader

from train.lightning_model_DMPNN import GraphMoleculeLightningDMPNN
from train.data_module import MoleculeDataModule # Assuming existing data module
from config import BATCH_SIZE, HIDDEN_DIM, NUM_LAYERS, LEARNING_RATE

def main():
    # torch.set_float32_matmul_precision('medium')
    # 1. Initialize Data Module
    # Points to your 'data/' folder as per repository structure
    data_module = MoleculeDataModule(
        data_dir="data", 
        batch_size=BATCH_SIZE,
        train_subset_size=1000 # Scaling to the 100k subset
    )

    # 2. Initialize the Lightning Model
    model = GraphMoleculeLightningDMPNN(
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        lr=LEARNING_RATE
    )

    # 3. Setup Checkpoints
    # Saves results to the 'results/' folder based on your requirement
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath="results/checkpoints_DMPNN/",
        filename="dmpnn-{epoch:02d}-{val_loss:.2f}",
        save_top_k=3,
        mode="min",
    )

    # 4. Initialize Trainer
    trainer = pl.Trainer(
        max_epochs=30,
        accelerator="auto",
        devices=1,
        callbacks=[checkpoint_callback],
        default_root_dir="results/DMPNN_logs"
    )

    # 5. Start Training
    print("🚀 Starting DMPNN training on 100k subset...")
    trainer.fit(model, datamodule=data_module)

if __name__ == "__main__":
    main()