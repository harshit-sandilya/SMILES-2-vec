import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from train.models.model_GIN import GraphMoleculeModelGIN
from config import *


class GraphMoleculeLightningGIN(pl.LightningModule):
    def __init__(
        self,
        hidden_dim=256,
        num_layers=5,
        ATOM_VOCAB_SIZE=ATOM_VOCAB_SIZE,
        BOND_VOCAB_SIZE=BOND_VOCAB_SIZE,
        lr=1e-4,
        weight_decay=1e-5,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = GraphMoleculeModelGIN(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            ATOM_VOCAB_SIZE=ATOM_VOCAB_SIZE,
            BOND_VOCAB_SIZE=BOND_VOCAB_SIZE,
        )

        self.lr = lr
        self.weight_decay = weight_decay

    # --------------------------------------------------
    # Forward (delegates to model)
    # --------------------------------------------------
    def forward(self, batch):
        return self.model(batch)

    # --------------------------------------------------
    # Training step
    # --------------------------------------------------
    def training_step(self, batch, batch_idx):
    # Just forward to get embeddings
        atom_logits, bond_logits = self(batch)

    # Dummy loss: mean of logits (self-supervised embedding training)
        loss = atom_logits.mean()

        self.log("train_loss", loss, prog_bar=True)
        return loss

        return loss

    # --------------------------------------------------
    # Validation step
    # --------------------------------------------------
    def validation_step(self, batch, batch_idx):
        atom_logits, bond_logits = self(batch)
        loss = atom_logits.mean()
        self.log("val_loss", loss, prog_bar=True)
        return loss

    # --------------------------------------------------
    # Optimizer
    # --------------------------------------------------
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=5,
        )   

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }

    # --------------------------------------------------
    # Embedding extraction (for cosine similarity)
    # --------------------------------------------------
    def get_embedding(self, batch):
        return self.model.get_embedding(batch)
