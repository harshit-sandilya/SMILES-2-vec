import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric import nn as gnn


class GraphMoleculeModel(nn.Module):
    def __init__(
        self,
        hidden_dim,
        num_layers,
        num_heads,
        ATOM_VOCAB_SIZE,
        BOND_VOCAB_SIZE,
    ):
        super().__init__()
        self.atom_embedder = nn.Embedding(ATOM_VOCAB_SIZE, hidden_dim)
        self.bond_embedder = nn.Embedding(BOND_VOCAB_SIZE, hidden_dim)
        self.gnn_layers = nn.ModuleList()
        for _ in range(num_layers):
            layer = gnn.GATv2Conv(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                heads=num_heads,
                concat=False,
                edge_dim=hidden_dim,
                dropout=0.1,
            )
            self.gnn_layers.append(layer)
        self.pool = gnn.aggr.AttentionalAggregation(gate_nn=nn.Linear(hidden_dim, 1))
        self.predict_atom = nn.Linear(hidden_dim, ATOM_VOCAB_SIZE)
        self.predict_bond = nn.Linear(hidden_dim * 2, BOND_VOCAB_SIZE)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        edge_index = edge_index
        edge_attr = edge_attr
        x = self.atom_embedder(x.squeeze())
        edge_attr = self.bond_embedder(edge_attr)
        for layer in self.gnn_layers:
            x_update = layer(x, edge_index, edge_attr)
            x = x + F.relu(x_update)
        predicted_atom_logits = self.predict_atom(x)
        row, col = edge_index
        atom_pair_features = torch.cat([x[row], x[col]], dim=-1)
        predicted_bond_logits = self.predict_bond(atom_pair_features)
        return predicted_atom_logits, predicted_bond_logits

    def get_embedding(self, data):
        x, edge_index, edge_attr, batch_idx = (
            data.x,
            data.edge_index,
            data.edge_attr,
            data.batch,
        )
        x = x
        edge_index = edge_index
        edge_attr = edge_attr
        batch_idx = batch_idx
        x = self.atom_embedder(x.squeeze())
        edge_attr = self.bond_embedder(edge_attr)
        for layer in self.gnn_layers:
            x_update = layer(x, edge_index, edge_attr)
            x = x + F.relu(x_update)
        molecule_embedding = self.pool(x, batch_idx)
        return molecule_embedding
