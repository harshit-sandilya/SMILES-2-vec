import pytorch_lightning as pl
import torch.nn.functional as F
import torch
from train.models.model_GCN import GraphMoleculeGCN
from config import ATOM_VOCAB_SIZE, HIDDEN_DIM


class GraphMoleculeLightningGCN(pl.LightningModule):
    def __init__(self, hidden_dim, num_layers, lr=1e-4, margin=-0.1):
        super().__init__()
        self.save_hyperparameters()

        self.model = GraphMoleculeGCN(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            ATOM_VOCAB_SIZE=ATOM_VOCAB_SIZE,
        )

        self.recon_loss = torch.nn.CrossEntropyLoss()

        # Projection head: maps pooled embeddings into a space specifically
        # optimized for distinction. Keeps the encoder free to serve both
        # recon and distinction without conflict.
        # FIX BUG 1: This is now actually called in _compute_distinction_loss.
        self.projector = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim // 2),
        )

        self.lr = lr
        # FIX BUG 5: margin defines the maximum allowed pairwise cosine similarity.
        # Any pair with sim > margin contributes to the loss.
        # Default -0.1 means we want all pairs to have sim <= -0.1 (slightly anti-correlated).
        self.margin = margin

    def forward(self, batch):
        return self.model(batch)

    # ─────────────────────────────────────────────
    # Shared helper: distinction loss
    # ─────────────────────────────────────────────
    def _compute_distinction_loss(self, batch):
        """
        Extracts graph embeddings, projects them, and computes a
        margin-clamped pairwise similarity loss.

        FIX BUG 1: projector is now actually used here.
        FIX BUG 5: F.relu clamps the loss so it can never go negative.
                   Loss is 0 when all pairwise sims are already <= margin.
        """
        embeddings = self.model.get_graph_embedding(batch)          # (B, hidden_dim)

        # FIX BUG 1: pass through projector before computing similarity.
        # The projector gives the model a dedicated low-dim space to spread
        # embeddings in, without fighting the recon objective in the encoder.
        projected   = self.projector(embeddings)                    # (B, hidden_dim//2)
        normalized  = F.normalize(projected, p=2, dim=1)           # unit vectors

        # Pairwise cosine similarity matrix
        sim_matrix  = torch.mm(normalized, normalized.t())         # (B, B)

        # Mask out the diagonal (self-similarity = 1.0, not useful)
        mask        = torch.eye(sim_matrix.size(0), device=self.device).bool()
        off_diag_sim = sim_matrix.masked_select(~mask)             # (B*B - B,)

        # FIX BUG 5: margin-clamped loss.
        #   relu(sim - margin) is:
        #     0         when sim <= margin   (pair is already spread enough)
        #     sim-margin when sim >  margin  (pair is too similar, penalize it)
        # This prevents the loss from going negative and gives clear gradient
        # only to pairs that actually violate the margin.
        loss = F.relu(off_diag_sim - self.margin).mean()

        return loss

    # ─────────────────────────────────────────────
    # Training
    # ─────────────────────────────────────────────
    def training_step(self, batch, batch_idx):
        # 1. Atom reconstruction loss (node-level classification)
        logits      = self(batch)
        target      = batch.x.squeeze()
        loss_recon  = self.recon_loss(logits, target)

        # 2. Embedding distinction loss (graph-level diversity)
        loss_distinction = self._compute_distinction_loss(batch)

        # Combined loss
        total_loss  = loss_recon + 0.5 * loss_distinction

        self.log("train_loss",        total_loss,       on_step=True, on_epoch=True)
        self.log("train_recon_loss",  loss_recon,       on_step=True, on_epoch=True)
        self.log("train_distinction", loss_distinction, on_step=True, on_epoch=True)

        return total_loss

    # ─────────────────────────────────────────────
    # Validation
    # ─────────────────────────────────────────────
    def validation_step(self, batch, batch_idx):
        # FIX BUG 4: val_loss now includes distinction_loss.
        # Previously only recon_loss was monitored, so EarlyStopping
        # could stop the model while embeddings were still collapsed.
        logits      = self(batch)
        target      = batch.x.squeeze()
        loss_recon  = self.recon_loss(logits, target)

        loss_distinction = self._compute_distinction_loss(batch)

        val_loss    = loss_recon + 0.5 * loss_distinction

        self.log("val_loss",           val_loss,         prog_bar=True)
        self.log("val_recon_loss",     loss_recon)
        self.log("val_distinction",    loss_distinction)

        return val_loss

    # ─────────────────────────────────────────────
    # Optimizer
    # ─────────────────────────────────────────────
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    # ─────────────────────────────────────────────
    # Embedding extraction (inference)
    # ─────────────────────────────────────────────
    def get_graph_embedding(self, batch):
        # Returns the pooled hidden-state embeddings (NOT projected).
        # The projector is only for training-time distinction loss;
        # the raw pooled embeddings are richer for downstream use.
        return self.model.get_graph_embedding(batch)
