import torch
import math
import pytorch_lightning as pl
import torch.nn.functional as F
from torch.optim import AdamW

from train.models.model_GATv2 import GraphMoleculeModelGATv2
from config_GATv2 import (
    ATOM_VOCAB_SIZE,
    BOND_VOCAB_SIZE,
    EMBEDDING_DIM,
    PROPERTY_LOSS_WEIGHT,
)


class GraphMoleculeLightningGATv2(pl.LightningModule):
    def __init__(
        self,
        hidden_dim=512,
        num_layers=6,
        num_heads=8,
        lr=2e-4,
        weight_decay=1e-5,
        warmup_steps=2000,
        property_loss_weight=PROPERTY_LOSS_WEIGHT,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = GraphMoleculeModelGATv2(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            ATOM_VOCAB_SIZE=ATOM_VOCAB_SIZE,
            BOND_VOCAB_SIZE=BOND_VOCAB_SIZE,
            embedding_dim=EMBEDDING_DIM,
            num_mol_props=4,
        )

        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.property_loss_weight = property_loss_weight

    # --------------------------------------------------
    # Forward
    # --------------------------------------------------
    def forward(self, batch):
        return self.model(batch)

    # --------------------------------------------------
    # Compute loss
    # --------------------------------------------------
    def _compute_loss(self, batch):
        atom_logits, bond_logits, graph_emb, predicted_props = self(batch)

        # ── Reconstruction losses ──
        atom_targets = batch.y_atoms.long()
        bond_targets = batch.y_bonds.long()

        atom_loss = F.cross_entropy(atom_logits, atom_targets, ignore_index=-1)
        bond_loss = F.cross_entropy(bond_logits, bond_targets, ignore_index=-1)
        recon_loss = atom_loss + bond_loss

        # ── Property prediction loss ──
        # Batch.from_data_list() concatenates each graph's mol_props [4] into
        # a flat [B*4] tensor. Reshape to [B, 4] before MSE so shapes match
        # predicted_props which is [B, 4] from the property head.
        mol_props = batch.mol_props.float().view(-1, 4)  # [B*4] → [B, 4]
        prop_loss = F.mse_loss(predicted_props, mol_props)

        total_loss = recon_loss + self.property_loss_weight * prop_loss

        return total_loss, recon_loss, atom_loss, bond_loss, prop_loss

    # --------------------------------------------------
    # Training step
    # --------------------------------------------------
    def training_step(self, batch, batch_idx):
        total_loss, recon_loss, atom_loss, bond_loss, prop_loss = self._compute_loss(
            batch
        )

        bs = batch.num_graphs
        self.log("train_loss", total_loss, prog_bar=True, batch_size=bs)
        self.log("train_recon_loss", recon_loss, prog_bar=False, batch_size=bs)
        self.log("train_atom_loss", atom_loss, prog_bar=False, batch_size=bs)
        self.log("train_bond_loss", bond_loss, prog_bar=False, batch_size=bs)
        self.log("train_prop_loss", prop_loss, prog_bar=True, batch_size=bs)

        return total_loss

    # --------------------------------------------------
    # Validation step
    # --------------------------------------------------
    def validation_step(self, batch, batch_idx):
        total_loss, recon_loss, atom_loss, bond_loss, prop_loss = self._compute_loss(
            batch
        )

        bs = batch.num_graphs
        self.log("val_loss", total_loss, prog_bar=True, batch_size=bs)
        self.log("val_recon_loss", recon_loss, prog_bar=False, batch_size=bs)
        self.log("val_atom_loss", atom_loss, prog_bar=False, batch_size=bs)
        self.log("val_bond_loss", bond_loss, prog_bar=False, batch_size=bs)
        self.log("val_prop_loss", prop_loss, prog_bar=False, batch_size=bs)

        return total_loss

    # --------------------------------------------------
    # Test step
    # --------------------------------------------------
    def test_step(self, batch, batch_idx):
        total_loss, recon_loss, atom_loss, bond_loss, prop_loss = self._compute_loss(
            batch
        )

        bs = batch.num_graphs
        self.log("test_loss", total_loss, prog_bar=True, batch_size=bs)
        self.log("test_recon_loss", recon_loss, prog_bar=False, batch_size=bs)
        self.log("test_atom_loss", atom_loss, prog_bar=False, batch_size=bs)
        self.log("test_bond_loss", bond_loss, prog_bar=False, batch_size=bs)
        self.log("test_prop_loss", prop_loss, prog_bar=False, batch_size=bs)

        return total_loss

    # --------------------------------------------------
    # Optimizer with linear warmup + cosine decay
    # --------------------------------------------------
    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        total_steps = self.trainer.estimated_stepping_batches

        def lr_lambda(current_step: int) -> float:
            if current_step < self.warmup_steps:
                return float(current_step) / float(max(1, self.warmup_steps))
            progress = float(current_step - self.warmup_steps) / float(
                max(1, total_steps - self.warmup_steps)
            )
            return max(0.01, 0.5 * (1.0 + math.cos(math.pi * progress)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    # --------------------------------------------------
    # Embedding API
    # --------------------------------------------------
    def get_embedding(self, batch):
        return self.model.get_embedding(batch)

    def get_graph_embedding(self, batch):
        return self.model.get_graph_embedding(batch)