import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool, global_max_pool

class DMPNNLayer(nn.Module):
    def __init__(self, hidden_dim):
        super(DMPNNLayer, self).init__()
        self.W_m = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, h_edges, line_graph_edge_index):
        # Directed message passing: edges pass messages to adjacent edges
        # excluding the reverse direction of the current bond.
        source_edges, target_edges = line_graph_edge_index
        
        # Message aggregation
        messages = torch.zeros_like(h_edges)
        messages.index_add_(0, target_edges, h_edges[source_edges])
        
        # Update edge hidden states
        h_edges = F.relu(self.W_m(messages))
        return h_edges

class GraphMoleculeDMPNN(nn.Module):
    def __init__(self, hidden_dim, num_layers, ATOM_VOCAB_SIZE):
        super(GraphMoleculeDMPNN, self).__init__()
        self.atom_embedder = nn.Embedding(ATOM_VOCAB_SIZE, hidden_dim)
        # Assuming bond features are simplified for now; 
        # normally DMPNN uses bond types (single, double, etc.)
        self.W_i = nn.Linear(hidden_dim, hidden_dim) 

        self.layers = nn.ModuleList([DMPNNLayer(hidden_dim) for _ in range(num_layers)])
        
        self.W_a = nn.Linear(hidden_dim * 2, hidden_dim) # Atom update
        self.predict_atom = nn.Linear(hidden_dim, ATOM_VOCAB_SIZE)

    def encode(self, data):
        x, edge_index, line_graph_edge_index, batch = \
            data.x, data.edge_index, data.line_graph_edge_index, data.batch
        
        # Initial atom features
        h_atoms = self.atom_embedder(x.squeeze())
        
        # Initial edge features (simplification: based on source atom)
        h_edges = F.relu(self.W_i(h_atoms[edge_index[0]]))

        # Message Passing on Directed Edges
        for layer in self.layers:
            h_edges = layer(h_edges, line_graph_edge_index)

        # Aggregate edge messages to update atoms
        m_atoms = torch.zeros_like(h_atoms)
        m_atoms.index_add_(0, edge_index[1], h_edges)
        
        # Concatenate original atom features with aggregated bond messages
        h_atoms = F.relu(self.W_a(torch.cat([h_atoms, m_atoms], dim=1)))

        return h_atoms, batch

    def forward(self, data):
        h_atoms, _ = self.encode(data)
        return self.predict_atom(h_atoms)

    def get_graph_embedding(self, data):
        h_atoms, batch = self.encode(data)
        # Hybrid pooling for better distinction as we did for GCN
        return global_add_pool(h_atoms, batch) + global_max_pool(h_atoms, batch)