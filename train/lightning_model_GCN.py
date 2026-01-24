import pytorch_lightning as pl
import torch.nn.functional as F
import torch

from train.models.model_GCN import GraphMoleculeGCN
from config import ATOM_VOCAB_SIZE


class GraphMoleculeLightningGCN(pl.LightningModule):
    def __init__(self, hidden_dim, num_layers, lr=1e-4):
        super().__init__()
        self.save_hyperparameters()

        self.model = GraphMoleculeGCN(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            ATOM_VOCAB_SIZE=ATOM_VOCAB_SIZE,
        )

        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.lr = lr

    def forward(self, batch):
        return self.model(batch)

    def training_step(self, batch, batch_idx):
        logits = self(batch)
        target = batch.x.squeeze()
        loss = self.loss_fn(logits, target)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        logits = self(batch)
        target = batch.x.squeeze()
        loss = self.loss_fn(logits, target)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    # for embeddings
    def get_graph_embedding(self, batch):
        return self.model.get_graph_embedding(batch)

