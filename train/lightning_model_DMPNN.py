import pytorch_lightning as pl
import torch.nn.functional as F
import torch
from train.models.model_DMPNN import GraphMoleculeDMPNN
from config import ATOM_VOCAB_SIZE


class GraphMoleculeLightningDMPNN(pl.LightningModule):
    def __init__(self, hidden_dim, num_layers, lr=1e-4):
        super().__init__()
        self.save_hyperparameters()

        self.model = GraphMoleculeDMPNN(
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
        loss_recon = self.loss_fn(logits, target)

        preds = torch.argmax(logits, dim=1)
        accuracy = (preds == target).float().mean()

        embeddings = self.get_graph_embedding(batch)
        normalized_embeddings = F.normalize(embeddings, p=2, dim=1)
        
        sim_matrix = torch.mm(normalized_embeddings, normalized_embeddings.t())
        
        mask = torch.eye(sim_matrix.size(0), device=self.device).bool()
        off_diag_sim = sim_matrix.masked_select(~mask)
        loss_distinction = off_diag_sim.mean()
        
        avg_similarity = off_diag_sim.mean()
        max_similarity = off_diag_sim.max()
        min_similarity = off_diag_sim.min()

        total_loss = loss_recon + 0.5 * loss_distinction
        
        self.log("train_loss", total_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_recon_loss", loss_recon, on_step=False, on_epoch=True)
        self.log("train_distinction_loss", loss_distinction, on_step=False, on_epoch=True)
        self.log("train_accuracy", accuracy, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_avg_similarity", avg_similarity, on_step=False, on_epoch=True)
        self.log("train_max_similarity", max_similarity, on_step=False, on_epoch=True)
        self.log("train_min_similarity", min_similarity, on_step=False, on_epoch=True)
        
        return total_loss

    def validation_step(self, batch, batch_idx):
        logits = self(batch)
        target = batch.x.squeeze()
        loss_recon = self.loss_fn(logits, target)
        
        preds = torch.argmax(logits, dim=1)
        accuracy = (preds == target).float().mean()
        
        embeddings = self.get_graph_embedding(batch)
        normalized_embeddings = F.normalize(embeddings, p=2, dim=1)
        sim_matrix = torch.mm(normalized_embeddings, normalized_embeddings.t())
        mask = torch.eye(sim_matrix.size(0), device=self.device).bool()
        off_diag_sim = sim_matrix.masked_select(~mask)
        
        avg_similarity = off_diag_sim.mean()
        loss_distinction = avg_similarity
        
        total_loss = loss_recon + 0.5 * loss_distinction
        
        self.log("val_loss", total_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_recon_loss", loss_recon, on_step=False, on_epoch=True)
        self.log("val_distinction_loss", loss_distinction, on_step=False, on_epoch=True)
        self.log("val_accuracy", accuracy, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_avg_similarity", avg_similarity, on_step=False, on_epoch=True)
        
        return total_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=3,
            verbose=True
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def get_graph_embedding(self, batch):
        return self.model.get_graph_embedding(batch)