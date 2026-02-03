import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool


class GraphMoleculeGCN(nn.Module):
    def __init__(self, hidden_dim, num_layers, ATOM_VOCAB_SIZE):
        super().__init__()

        self.atom_embedder = nn.Embedding(ATOM_VOCAB_SIZE, hidden_dim)
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        for _ in range(num_layers):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        self.predict_atom = nn.Linear(hidden_dim, ATOM_VOCAB_SIZE)

        # FIX BUG 3: After concatenating mean + max pooling, we get 2*hidden_dim.
        # This linear layer projects it back to hidden_dim so the rest of the
        # pipeline (projector, predict_atom, etc.) stays dimension-consistent.
        # Summing mean+max loses the max signal; concatenation preserves both.
        self.pool_proj = nn.Linear(hidden_dim * 2, hidden_dim)

    def encode(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.atom_embedder(x.squeeze())

        for conv, bn in zip(self.convs, self.bns):
            h = conv(x, edge_index)
            h = bn(h)
            h = F.relu(h)
            x = x + h  # Residual connection

        return x, batch

    def forward(self, data):
        x, batch = self.encode(data)
        logits = self.predict_atom(x)
        return logits

    def get_graph_embedding(self, data):
        x, batch = self.encode(data)

        mean_pool = global_mean_pool(x, batch)  # (num_graphs, hidden_dim)
        max_pool  = global_max_pool(x, batch)   # (num_graphs, hidden_dim)

        # FIX BUG 3: Concatenate then project, instead of summing.
        # Concatenation keeps both signals fully intact.
        # pool_proj learns how to optimally combine them.
        combined  = torch.cat([mean_pool, max_pool], dim=1)  # (num_graphs, 2*hidden_dim)
        graph_emb = F.relu(self.pool_proj(combined))         # (num_graphs, hidden_dim)

        return graph_emb
