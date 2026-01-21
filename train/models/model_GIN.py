import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv
from torch_geometric.nn.aggr import AttentionalAggregation


class GraphMoleculeModelGIN(nn.Module):
    def __init__(
        self,
        hidden_dim,
        num_layers,
        ATOM_VOCAB_SIZE,
        BOND_VOCAB_SIZE,
    ):
        super().__init__()

        # ---------------- Embeddings ----------------
        self.atom_embedder = nn.Embedding(ATOM_VOCAB_SIZE, hidden_dim)
        self.bond_embedder = nn.Embedding(BOND_VOCAB_SIZE, hidden_dim)

        # ---------------- GIN layers ----------------
        self.gnn_layers = nn.ModuleList()

        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )

            conv = GINEConv(
                nn=mlp,
                edge_dim=hidden_dim,
                train_eps=True,
            )
            self.gnn_layers.append(conv)

        # ---------------- Pooling ----------------
        self.pool = AttentionalAggregation(
            gate_nn=nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )
        )

        # ---------------- Prediction heads ----------------
        self.predict_atom = nn.Linear(hidden_dim, ATOM_VOCAB_SIZE)
        self.predict_bond = nn.Linear(hidden_dim * 2, BOND_VOCAB_SIZE)

    # ====================================================
    # Forward pass (used during masked training)
    # ====================================================
    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        # Embed atoms & bonds
        x = self.atom_embedder(x.squeeze())
        edge_attr = self.bond_embedder(edge_attr)

        # GIN message passing
        for conv in self.gnn_layers:
            x = x + F.relu(conv(x, edge_index, edge_attr))

        # Atom prediction
        predicted_atom_logits = self.predict_atom(x)

        # Bond prediction
        row, col = edge_index
        atom_pair_features = torch.cat([x[row], x[col]], dim=-1)
        predicted_bond_logits = self.predict_bond(atom_pair_features)

        return predicted_atom_logits, predicted_bond_logits

    # ====================================================
    # Embedding extraction (original API)
    # ====================================================
    def get_embedding(self, data):
        x, edge_index, edge_attr, batch_idx = (
            data.x,
            data.edge_index,
            data.edge_attr,
            data.batch,
        )

        x = self.atom_embedder(x.squeeze())
        edge_attr = self.bond_embedder(edge_attr)

        for conv in self.gnn_layers:
            x = x + F.relu(conv(x, edge_index, edge_attr))

        molecule_embedding = self.pool(x, batch_idx)
        return molecule_embedding

    # ====================================================
    # Graph-level embedding (GCN/GAT-compatible API) ✅ NEW
    # ====================================================
    def get_graph_embedding(self, data):
        """
        Alias for get_embedding() to keep a unified interface
        across GCN / GIN / GAT models.
        """
        return self.get_embedding(data)
