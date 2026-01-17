import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from torch_geometric.nn import global_mean_pool
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

        # ✅ FIX 2: normalization for GIN stability
        self.graph_norm = torch.nn.LayerNorm(hidden_dim)

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
        atom_logits, bond_logits = self(batch)

        # Dummy self-supervised loss
        loss = atom_logits.mean()

        self.log("train_loss", loss, prog_bar=True)
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
    # Embedding extraction (FIXED)
    # --------------------------------------------------
    def get_embedding(self, batch):
        """
        Returns graph-level embeddings.
        Shape: (num_graphs, hidden_dim)
        """

        x, edge_index, batch_idx = batch.x, batch.edge_index, batch.batch

        # 1️⃣ Node-level GIN embeddings
        x = self.model.gnn(x, edge_index)

        # 2️⃣ Global pooling (ABSOLUTELY REQUIRED)
        x = global_mean_pool(x, batch_idx)

        # 3️⃣ FIX 2: normalize pooled graph embeddings
        x = self.graph_norm(x)

        # 4️⃣ FIX 3: numerical safety
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        # 5️⃣ L2 normalization (embedding space stability)
        x = F.normalize(x, p=2, dim=1)

        return x
