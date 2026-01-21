import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool


class GraphMoleculeGCN(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_layers: int,
        ATOM_VOCAB_SIZE: int,
    ):
        super().__init__()

        # -----------------------------
        # Atom embedding
        # -----------------------------
        self.atom_embedder = nn.Embedding(
            ATOM_VOCAB_SIZE,
            hidden_dim,
        )

        # -----------------------------
        # GCN layers
        # -----------------------------
        self.gcn_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()

        for _ in range(num_layers):
            self.gcn_layers.append(
                GCNConv(hidden_dim, hidden_dim)
            )
            self.norm_layers.append(
                nn.LayerNorm(hidden_dim)
            )

        # -----------------------------
        # Graph pooling
        # -----------------------------
        self.pool = global_mean_pool

    # --------------------------------------------------
    # Shared encoder
    # --------------------------------------------------
    def _encode(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # x shape: [num_nodes, 1] → squeeze
        x = self.atom_embedder(x.squeeze(-1))

        for conv, norm in zip(self.gcn_layers, self.norm_layers):
            x_update = conv(x, edge_index)
            x = norm(x + F.relu(x_update))  # residual GCN

        return x, batch

    # --------------------------------------------------
    # Graph-level embedding
    # --------------------------------------------------
    def get_graph_embedding(self, data):
        x, batch = self._encode(data)
        graph_emb = self.pool(x, batch)
        return graph_emb
