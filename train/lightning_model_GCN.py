import pytorch_lightning as pl
import torch

from train.models.model_GCN import GraphMoleculeGCN
from config import ATOM_VOCAB_SIZE

class GraphMoleculeLightningGCN(pl.LightningModule):
    def __init__(
        self,
        hidden_dim: int,
        num_layers: int,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = GraphMoleculeGCN(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            ATOM_VOCAB_SIZE=ATOM_VOCAB_SIZE,
        )

    # --------------------------------------------------
    # Forward (optional, not used for training)
    # --------------------------------------------------
    def forward(self, batch):
        return self.model.get_graph_embedding(batch)

    # --------------------------------------------------
    # Inference helper (explicit & clean)
    # --------------------------------------------------
    def get_graph_embedding(self, batch):
        return self.model.get_graph_embedding(batch)

    # --------------------------------------------------
    # No optimizers needed (inference-only)
    # --------------------------------------------------
    def configure_optimizers(self):
        return None
