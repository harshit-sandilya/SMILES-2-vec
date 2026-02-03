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
    # FIX #1: proper masked reconstruction loss
    # --------------------------------------------------
    def _compute_loss(self, batch):
        atom_logits, bond_logits = self(batch)

        # --- Atom loss (only on masked positions) ---
        # y_atoms == -1 means "not masked"; ignore those.
        atom_loss = F.cross_entropy(
            atom_logits,           # [total_atoms, ATOM_VOCAB_SIZE]
            batch.y_atoms,         # [total_atoms]  (-1 = ignore)
            ignore_index=-1,
        )

        # --- Bond loss (only on masked positions) ---
        # y_bonds == -1 means "not masked"; ignore those.
        bond_loss = F.cross_entropy(
            bond_logits,           # [total_edges, BOND_VOCAB_SIZE]
            batch.y_bonds,         # [total_edges]  (-1 = ignore)
            ignore_index=-1,
        )

        # Combined loss (equal weight; can tune later)
        loss = atom_loss + bond_loss
        return loss, atom_loss, bond_loss

    # --------------------------------------------------
    # Training step
    # --------------------------------------------------
    def training_step(self, batch, batch_idx):
        loss, atom_loss, bond_loss = self._compute_loss(batch)
        self.log("train_loss",      loss,       prog_bar=True,  batch_size=batch.num_graphs)
        self.log("train_atom_loss", atom_loss,  prog_bar=False, batch_size=batch.num_graphs)
        self.log("train_bond_loss", bond_loss,  prog_bar=False, batch_size=batch.num_graphs)
        return loss

    # --------------------------------------------------
    # Validation
    # --------------------------------------------------
    def validation_step(self, batch, batch_idx):
        loss, atom_loss, bond_loss = self._compute_loss(batch)
        self.log("val_loss",      loss,       prog_bar=True,  batch_size=batch.num_graphs)
        self.log("val_atom_loss", atom_loss,  prog_bar=False, batch_size=batch.num_graphs)
        self.log("val_bond_loss", bond_loss,  prog_bar=False, batch_size=batch.num_graphs)
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
    # Embedding API (for generate_embeddings)
    # ==================================================
    def get_embedding(self, batch):
        return self.model.get_embedding(batch)

    def get_graph_embedding(self, batch):
        return self.model.get_graph_embedding(batch)
