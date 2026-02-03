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

        # ---------------- GIN layers + LayerNorm ----------------  # FIX #6
        self.gnn_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()

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
            self.layer_norms.append(nn.LayerNorm(hidden_dim))

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
    # Shared message-passing core                         # FIX #7
    # ====================================================
    def _message_passing(self, x, edge_index, edge_attr):
        """
        Embed raw integer features, run GIN layers with residual
        connections and layer norm, return final node embeddings.
        """
        # FIX #8: use .view(-1) instead of .squeeze() to avoid
        # collapsing batch dim when batch_size=1 or num_atoms=1.
        # Also enforce long dtype so nn.Embedding doesn't crash.
        x = self.atom_embedder(x.view(-1).long())
        edge_attr = self.bond_embedder(edge_attr.long())

        for conv, ln in zip(self.gnn_layers, self.layer_norms):
            x = ln(x + F.relu(conv(x, edge_index, edge_attr)))   # FIX #6

        return x

    # ====================================================
    # Forward pass (used during masked training)
    # ====================================================
    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        x = self._message_passing(x, edge_index, edge_attr)      # FIX #7

        # Atom prediction
        predicted_atom_logits = self.predict_atom(x)

        # Bond prediction
        row, col = edge_index
        atom_pair_features = torch.cat([x[row], x[col]], dim=-1)
        predicted_bond_logits = self.predict_bond(atom_pair_features)

        return predicted_atom_logits, predicted_bond_logits

    # ====================================================
    # Embedding extraction
    # ====================================================
    def get_embedding(self, data):
        x, edge_index, edge_attr, batch_idx = (
            data.x,
            data.edge_index,
            data.edge_attr,
            data.batch,
        )

        x = self._message_passing(x, edge_index, edge_attr)      # FIX #7

        molecule_embedding = self.pool(x, batch_idx)
        return molecule_embedding

    # ====================================================
    # Graph-level embedding (unified API across GCN/GIN/GAT)
    # ====================================================
    def get_graph_embedding(self, data):
        return self.get_embedding(data)
