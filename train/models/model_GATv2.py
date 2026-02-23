import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_add_pool, global_max_pool, global_mean_pool


class GraphMoleculeModelGATv2(nn.Module):
    def __init__(
        self,
        hidden_dim,
        num_layers,
        num_heads,
        ATOM_VOCAB_SIZE,
        BOND_VOCAB_SIZE,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        # ---------------- Embeddings ----------------
        self.atom_embedder = nn.Embedding(ATOM_VOCAB_SIZE, hidden_dim)
        self.bond_embedder = nn.Embedding(BOND_VOCAB_SIZE, hidden_dim)

        # ---------------- GATv2 layers ----------------
        self.gnn_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()

        for _ in range(num_layers):
            conv = GATv2Conv(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                heads=num_heads,
                edge_dim=hidden_dim,
                concat=False,  # Average heads instead of concat
                dropout=0.1,   # Attention dropout
            )
            self.gnn_layers.append(conv)
            self.layer_norms.append(nn.LayerNorm(hidden_dim))

        # ---------------- SMART POOLING ----------------
        # Triple pooling: sum + max + mean → MLP projection
        # This creates a 3*hidden_dim dimensional embedding
        # Then project back to hidden_dim for final output
        self.pool_projection = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

        # ---------------- Prediction heads ----------------
        self.predict_atom = nn.Linear(hidden_dim, ATOM_VOCAB_SIZE)
        self.predict_bond = nn.Linear(hidden_dim * 2, BOND_VOCAB_SIZE)

    # ====================================================
    # Shared message-passing core
    # ====================================================
    def _message_passing(self, x, edge_index, edge_attr):
        """
        Embed raw integer features, run GATv2 layers with residual
        connections and layer norm, return final node embeddings.
        """
        x = self.atom_embedder(x.view(-1).long())
        edge_attr = self.bond_embedder(edge_attr.view(-1).long())

        for conv, ln in zip(self.gnn_layers, self.layer_norms):
            x_update = conv(x, edge_index, edge_attr)
            x = ln(x + F.relu(x_update))  # Residual connection

        return x

    # ====================================================
    # Forward pass (used during masked training)
    # ====================================================
    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        x = self._message_passing(x, edge_index, edge_attr)

        # Atom prediction
        predicted_atom_logits = self.predict_atom(x)

        # Bond prediction
        row, col = edge_index
        atom_pair_features = torch.cat([x[row], x[col]], dim=-1)
        predicted_bond_logits = self.predict_bond(atom_pair_features)

        return predicted_atom_logits, predicted_bond_logits

    # ====================================================
    # SMART POOLING - Multiple strategies combined
    # ====================================================
    def get_embedding(self, data):
        x, edge_index, edge_attr, batch_idx = (
            data.x,
            data.edge_index,
            data.edge_attr,
            data.batch,
        )

        x = self._message_passing(x, edge_index, edge_attr)

        # Triple pooling strategy:
        # 1. Sum pooling (global_add_pool) - captures total atomic properties
        # 2. Max pooling (global_max_pool) - captures most salient features
        # 3. Mean pooling (global_mean_pool) - captures average properties
        sum_pool = global_add_pool(x, batch_idx)
        max_pool = global_max_pool(x, batch_idx)
        mean_pool = global_mean_pool(x, batch_idx)
        
        # Concatenate all three: 3*hidden_dim
        combined = torch.cat([sum_pool, max_pool, mean_pool], dim=-1)
        
        # Project back to hidden_dim through MLP
        # This learns optimal combination of pooling strategies
        molecule_embedding = self.pool_projection(combined)
        
        # L2 normalize for better cosine similarity properties
        # This ensures embeddings lie on unit hypersphere
        molecule_embedding = F.normalize(molecule_embedding, p=2, dim=-1)
        
        return molecule_embedding

    # ====================================================
    # Graph-level embedding (unified API)
    # ====================================================
    def get_graph_embedding(self, data):
        return self.get_embedding(data)