import os

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.nn import CrossEntropyLoss

from config import *
from data_module import MoleculeDataModule
from model import GraphMoleculeModel


class GraphMoleculeLightning(pl.LightningModule):
    def __init__(
        self,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        learning_rate: float = 1e-4,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = GraphMoleculeModel(
            hidden_dim, num_layers, num_heads, ATOM_VOCAB_SIZE, BOND_VOCAB_SIZE
        )
        self.loss_fn_atom = CrossEntropyLoss(ignore_index=-1)
        self.loss_fn_bond = CrossEntropyLoss(ignore_index=-1)

    def training_step(self, batch, batch_idx):
        atom_logits, bond_logits = self.model(batch)
        atom_loss = self.loss_fn_atom(atom_logits, batch.y_atoms)
        masked_bond_mask = batch.y_bonds != -1
        bond_loss = torch.tensor(0.0, device=self.device)
        if masked_bond_mask.sum() > 0:
            bond_loss = self.loss_fn_bond(
                bond_logits[masked_bond_mask], batch.y_bonds[masked_bond_mask]
            )
        if torch.isnan(bond_loss):
            bond_loss = torch.tensor(0.0, device=self.device)
        total_loss = atom_loss + bond_loss
        self.log("train_loss", total_loss, prog_bar=True, sync_dist=True)
        self.log("train_atom_loss", atom_loss, prog_bar=True, sync_dist=True)
        self.log("train_bond_loss", bond_loss, prog_bar=True, sync_dist=True)
        return total_loss

    def validation_step(self, batch, batch_idx):
        atom_logits, bond_logits = self.model(batch)
        atom_loss = self.loss_fn_atom(atom_logits, batch.y_atoms)
        masked_bond_mask = batch.y_bonds != -1
        bond_loss = torch.tensor(0.0, device=self.device)
        if masked_bond_mask.sum() > 0:
            bond_loss = self.loss_fn_bond(
                bond_logits[masked_bond_mask], batch.y_bonds[masked_bond_mask]
            )
        if torch.isnan(bond_loss):
            bond_loss = torch.tensor(0.0, device=self.device)
        total_loss = atom_loss + bond_loss
        self.log("val_loss", total_loss, prog_bar=True, sync_dist=True)
        self.log("val_atom_loss", atom_loss, prog_bar=True, sync_dist=True)
        self.log("val_bond_loss", bond_loss, prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.hparams.learning_rate, weight_decay=1e-5
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.max_epochs, eta_min=1e-6
        )
        return [optimizer], [scheduler]


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
        dirpath="checkpoints/",
        filename="model-{epoch:02d}",
        every_n_epochs=2,
        save_top_k=-1,
    )
    tensorboard_logger = TensorBoardLogger(
        save_dir="lightning_logs/", name="graph_molecule_model"
    )
    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        accelerator="cpu",
        callbacks=[checkpoint_callback],
        logger=tensorboard_logger,
        log_every_n_steps=10,
    )
    print("Starting training on CPU...")
    trainer.fit(model, datamodule=datamodule)
    print("Training completed.")
