import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool


class GraphMoleculeGCN(nn.Module):
    def __init__(self, hidden_dim, num_layers, ATOM_VOCAB_SIZE):
        super().__init__()

        self.atom_embedder = nn.Embedding(ATOM_VOCAB_SIZE, hidden_dim)

        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))

        self.pool = global_mean_pool

        # prediction head (for self-supervised training)
        self.predict_atom = nn.Linear(hidden_dim, ATOM_VOCAB_SIZE)

    def encode(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.atom_embedder(x.squeeze())

        for conv in self.convs:
            x = F.relu(conv(x, edge_index))

        return x, batch

    # used during training
    def forward(self, data):
        x, batch = self.encode(data)
        logits = self.predict_atom(x)
        return logits

    # used for embeddings
    def get_graph_embedding(self, data):
        x, batch = self.encode(data)
        return self.pool(x, batch)
