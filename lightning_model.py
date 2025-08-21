from model import GraphMoleculeModel
from torch.nn import CrossEntropyLoss
import torch
import pytorch_lightning as pl

from config import *


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
