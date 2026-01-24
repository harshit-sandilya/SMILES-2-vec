import torch
import pytorch_lightning as pl
import torch.nn.functional as F
from torch.optim import AdamW

from train.models.model_GIN import GraphMoleculeModelGIN
from config import ATOM_VOCAB_SIZE, BOND_VOCAB_SIZE


class GraphMoleculeLightningGIN(pl.LightningModule):
    def __init__(
        self,
        hidden_dim=256,
        num_layers=5,
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
    # Forward
    # --------------------------------------------------
    def forward(self, batch):
        return self.model(batch)

    # --------------------------------------------------
    # Training step
    # --------------------------------------------------
    def training_step(self, batch, batch_idx):
        atom_logits, bond_logits = self(batch)
        loss = atom_logits.mean()
        self.log("train_loss", loss, prog_bar=True, batch_size=batch.num_graphs)
        return loss

    # --------------------------------------------------
    # Validation
    # --------------------------------------------------
    def validation_step(self, batch, batch_idx):
        atom_logits, bond_logits = self(batch)
        loss = atom_logits.mean()
        self.log("val_loss", loss, prog_bar=True, batch_size=batch.num_graphs)
        return loss

    # --------------------------------------------------
    # Optimizer
    # --------------------------------------------------
    def configure_optimizers(self):
        return AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

    # ==================================================
    # 🔥 CRITICAL: Embedding API (for generate_embeddings)
    # ==================================================
    def get_embedding(self, batch):
        return self.model.get_embedding(batch)

    def get_graph_embedding(self, batch):
        return self.model.get_graph_embedding(batch)
