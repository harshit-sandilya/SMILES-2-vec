import torch
import pytorch_lightning as pl
import torch.nn.functional as F
from torch.optim import AdamW

from train.models.model_GATv2 import GraphMoleculeModelGATv2
from config_GATv2 import ATOM_VOCAB_SIZE, BOND_VOCAB_SIZE


class GraphMoleculeLightningGATv2(pl.LightningModule):
    def __init__(
        self,
        hidden_dim=512,
        num_layers=6,
        num_heads=8,
        lr=2e-4,
        weight_decay=1e-5,
        use_contrastive=False,
        contrastive_weight=0.1,
        contrastive_temperature=0.07,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = GraphMoleculeModelGATv2(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            ATOM_VOCAB_SIZE=ATOM_VOCAB_SIZE,
            BOND_VOCAB_SIZE=BOND_VOCAB_SIZE,
        )

        self.lr = lr
        self.weight_decay = weight_decay
        self.use_contrastive = use_contrastive
        self.contrastive_weight = contrastive_weight
        self.contrastive_temperature = contrastive_temperature

    # --------------------------------------------------
    # Forward
    # --------------------------------------------------
    def forward(self, batch):
        return self.model(batch)

    # --------------------------------------------------
    # Contrastive Loss (Optional) - Encourages diversity
    # --------------------------------------------------
    def _contrastive_loss(self, batch):
        """
        Intra-batch contrastive loss to encourage diverse embeddings.
        Molecules in the same batch are treated as negatives.
        This prevents all embeddings from collapsing to similar values.
        """
        # Get graph embeddings
        embeddings = self.model.get_graph_embedding(batch)  # [batch_size, hidden_dim]
        
        # Embeddings are already L2-normalized in the model
        # Compute similarity matrix
        sim_matrix = torch.mm(embeddings, embeddings.t()) / self.contrastive_temperature
        
        # Mask out diagonal (self-similarity)
        mask = torch.eye(sim_matrix.size(0), device=self.device).bool()
        sim_matrix = sim_matrix.masked_fill(mask, float('-inf'))
        
        # Contrastive loss: minimize average pairwise similarity
        # (encourage embeddings to spread out)
        loss = -torch.logsumexp(sim_matrix, dim=1).mean()
        
        return loss

    # --------------------------------------------------
    # Compute Loss
    # --------------------------------------------------
    def _compute_loss(self, batch):
        atom_logits, bond_logits = self(batch)

        # --- Reconstruction losses ---
        atom_loss = F.cross_entropy(
            atom_logits,
            batch.y_atoms,
            ignore_index=-1,
        )

        bond_loss = F.cross_entropy(
            bond_logits,
            batch.y_bonds,
            ignore_index=-1,
        )

        reconstruction_loss = atom_loss + bond_loss
        
        # --- Optional contrastive loss ---
        if self.use_contrastive:
            contrastive_loss = self._contrastive_loss(batch)
            total_loss = reconstruction_loss + self.contrastive_weight * contrastive_loss
        else:
            contrastive_loss = torch.tensor(0.0, device=self.device)
            total_loss = reconstruction_loss

        return total_loss, reconstruction_loss, atom_loss, bond_loss, contrastive_loss

    # --------------------------------------------------
    # Training step
    # --------------------------------------------------
    def training_step(self, batch, batch_idx):
        total_loss, recon_loss, atom_loss, bond_loss, contrast_loss = self._compute_loss(batch)
        
        self.log("train_loss",            total_loss,  prog_bar=True,  batch_size=batch.num_graphs)
        self.log("train_recon_loss",      recon_loss,  prog_bar=False, batch_size=batch.num_graphs)
        self.log("train_atom_loss",       atom_loss,   prog_bar=False, batch_size=batch.num_graphs)
        self.log("train_bond_loss",       bond_loss,   prog_bar=False, batch_size=batch.num_graphs)
        
        if self.use_contrastive:
            self.log("train_contrast_loss", contrast_loss, prog_bar=True, batch_size=batch.num_graphs)
        
        return total_loss

    # --------------------------------------------------
    # Validation
    # --------------------------------------------------
    def validation_step(self, batch, batch_idx):
        total_loss, recon_loss, atom_loss, bond_loss, contrast_loss = self._compute_loss(batch)
        
        self.log("val_loss",            total_loss,  prog_bar=True,  batch_size=batch.num_graphs)
        self.log("val_recon_loss",      recon_loss,  prog_bar=False, batch_size=batch.num_graphs)
        self.log("val_atom_loss",       atom_loss,   prog_bar=False, batch_size=batch.num_graphs)
        self.log("val_bond_loss",       bond_loss,   prog_bar=False, batch_size=batch.num_graphs)
        
        if self.use_contrastive:
            self.log("val_contrast_loss", contrast_loss, prog_bar=False, batch_size=batch.num_graphs)
        
        return total_loss

    # --------------------------------------------------
    # Test
    # --------------------------------------------------
    def test_step(self, batch, batch_idx):
        total_loss, recon_loss, atom_loss, bond_loss, contrast_loss = self._compute_loss(batch)
        
        self.log("test_loss",            total_loss,  prog_bar=True,  batch_size=batch.num_graphs)
        self.log("test_recon_loss",      recon_loss,  prog_bar=False, batch_size=batch.num_graphs)
        self.log("test_atom_loss",       atom_loss,   prog_bar=False, batch_size=batch.num_graphs)
        self.log("test_bond_loss",       bond_loss,   prog_bar=False, batch_size=batch.num_graphs)
        
        if self.use_contrastive:
            self.log("test_contrast_loss", contrast_loss, prog_bar=False, batch_size=batch.num_graphs)
        
        return total_loss

    # --------------------------------------------------
    # Optimizer with warmup
    # --------------------------------------------------
    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        
        # Cosine annealing with warmup for better convergence
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=10,  # Restart every 10 epochs
            T_mult=2,
            eta_min=self.lr * 0.01,
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }

    # ==================================================
    # Embedding API
    # ==================================================
    def get_embedding(self, batch):
        return self.model.get_embedding(batch)

    def get_graph_embedding(self, batch):
        return self.model.get_graph_embedding(batch)