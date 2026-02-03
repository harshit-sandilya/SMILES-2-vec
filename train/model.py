import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric import nn as gnn
from torch_geometric.nn import global_mean_pool


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
        self.norm_layers = nn.ModuleList()

        for _ in range(num_layers):
            self.gnn_layers.append(
                gnn.GATv2Conv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    heads=num_heads,
                    concat=False,
                    edge_dim=hidden_dim,
                    dropout=0.1,
                )
            )
            # FIX 2: normalization per layer
            self.norm_layers.append(nn.LayerNorm(hidden_dim))

        # Graph pooling
        self.pool = gnn.aggr.AttentionalAggregation(
            gate_nn=nn.Linear(hidden_dim, 1)
        )

        # Prediction heads (unchanged)
        self.predict_atom = nn.Linear(hidden_dim, ATOM_VOCAB_SIZE)
        self.predict_bond = nn.Linear(hidden_dim * 2, BOND_VOCAB_SIZE)

    # --------------------------------------------------
    # Shared encoder (THE IMPORTANT PART)
    # --------------------------------------------------
    def _encode(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        x = self.atom_embedder(x.squeeze())
        edge_attr = self.bond_embedder(edge_attr)

        for layer, norm in zip(self.gnn_layers, self.norm_layers):
            x_update = layer(x, edge_index, edge_attr)
            x = norm(x + F.relu(x_update))  # 🔧 stabilized residual

        return x

    # --------------------------------------------------
    # Forward (used during training)
    # --------------------------------------------------
    def forward(self, data):
        x = self._encode(data)

        # Atom prediction
        predicted_atom_logits = self.predict_atom(x)

        # Bond prediction
        row, col = data.edge_index
        atom_pair_features = torch.cat([x[row], x[col]], dim=-1)
        predicted_bond_logits = self.predict_bond(atom_pair_features)

        return predicted_atom_logits, predicted_bond_logits
    
    def get_embedding(self, batch):
        """
        Returns a graph-level embedding for cosine similarity.
        """
        _, _, graph_emb = self.forward(batch)
        return graph_emb

        # If shape is [num_nodes, hidden_dim]
        # convert to one vector per molecule
        if atom_logits.dim() == 2:
            emb = atom_logits.mean(dim=0, keepdim=True)
        else:
            emb = atom_logits

        return emb

    # --------------------------------------------------
    # Graph-level embedding (used for visualization / similarity)
    # --------------------------------------------------
    def get_graph_embedding(self, data):
        x = self._encode(data)

        # Graph pooling
        graph_emb = self.pool(x, data.batch)

        return graph_emb
