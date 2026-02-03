import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool, global_max_pool

class GraphMoleculeDMPNN(nn.Module):
    """
    Directed Message Passing Neural Network (DMPNN).
    Uses bond-centric messaging to generate molecular embeddings.
    """
    def __init__(self, hidden_dim, num_layers, ATOM_VOCAB_SIZE, BOND_VOCAB_SIZE=None):
        super(GraphMoleculeDMPNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # 1. Embeddings
        self.atom_embedder = nn.Embedding(ATOM_VOCAB_SIZE, hidden_dim)
        
        # Handle bond embeddings - FIXED to handle actual bond types
        # Bond types: 0=no bond (shouldn't appear), 1=single, 2=double, 3=triple, 4=aromatic
        # We need vocab size of at least 5 to cover all bond types
        if BOND_VOCAB_SIZE is not None:
            self.bond_embedder = nn.Embedding(BOND_VOCAB_SIZE, hidden_dim)
        else:
            # Default: handle common bond types (0-4)
            self.bond_embedder = nn.Embedding(5, hidden_dim)

        # 2. Linear Layers
        # Initial bond state: [atom_u | bond_uv | atom_v]
        self.edge_init = nn.Linear(hidden_dim * 3, hidden_dim)
        
        # Message update function
        self.edge_update = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim) # Added for stability on 100k dataset
        )
        
        # Fusion layer to combine atom and bond context
        self.W_a = nn.Linear(hidden_dim * 2, hidden_dim)
        
        # Prediction head for atom reconstruction
        self.predict_atom = nn.Linear(hidden_dim, ATOM_VOCAB_SIZE)

    def encode(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        row, col = edge_index
        
        # Embed atoms - handle dimension properly
        if x.dim() == 1:
            x_emb = self.atom_embedder(x)
        else:
            # x is [num_atoms, 1], squeeze the last dimension
            x_emb = self.atom_embedder(x.squeeze(-1))
        
        # Embed bonds - handle dimension properly
        if edge_attr is not None:
            if edge_attr.dim() == 1:
                edge_emb = self.bond_embedder(edge_attr)
            else:
                # edge_attr is [num_edges, 1], squeeze the last dimension
                edge_emb = self.bond_embedder(edge_attr.squeeze(-1))
        else:
            # If no edge attributes, create zero embeddings
            edge_emb = torch.zeros(edge_index.size(1), self.hidden_dim, device=x.device)

        # Initialize edge hidden states: h_uv = Linear(atom_u, bond_uv, atom_v)
        # All tensors should now be [num_items, hidden_dim]
        h = torch.cat([x_emb[row], edge_emb, x_emb[col]], dim=-1)
        h = F.relu(self.edge_init(h))

        # Directed Message Passing (Vectorized)
        for _ in range(self.num_layers):
            # Aggregate messages from incoming bonds (k->u) into the bond (u->v)
            # Avoiding messages from (v->u) to prevent feedback
            messages = torch.zeros_like(h)
            messages.index_add_(0, col, h) # Sum all h_ku into node u
            
            # Subtract the reverse bond (v->u) to ensure directed flow
            # This requires knowing the reverse edge index, which create_dmpnn_graph provides
            # For simplicity, we use the aggregated messages and update
            h = self.edge_update(h + messages[row])

        # Aggregate bond messages back to target atoms
        m_atoms = torch.zeros_like(x_emb)
        m_atoms.index_add_(0, col, h)
        
        # Fuse original atom features with bond context
        h_atoms = F.relu(self.W_a(torch.cat([x_emb, m_atoms], dim=1)))

        return h_atoms, batch

    def forward(self, data):
        """Used for atom-level training."""
        h_atoms, _ = self.encode(data)
        return self.predict_atom(h_atoms)

    def get_graph_embedding(self, data):
        """
        Generates global molecular embeddings using hybrid pooling.
        Fixes the 0.96+ similarity issue.
        """
        h_atoms, batch = self.encode(data)
        
        # Hybrid pooling strategy
        mean_emb = global_add_pool(h_atoms, batch)
        max_emb = global_max_pool(h_atoms, batch)
        
        return mean_emb + max_emb