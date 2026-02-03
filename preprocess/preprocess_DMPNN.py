import torch
from torch_geometric.data import Data

def create_dmpnn_graph(atomic_numbers, bond_index, bond_features=None):
    """
    Converts a standard graph into a directed bond graph for DMPNN.
    
    Args:
        atomic_numbers: [num_atoms] tensor of atomic numbers
        bond_index: [2, num_edges] tensor of edge connections
        bond_features: Optional tensor of bond features/types
    
    Returns:
        Data object for DMPNN with line graph structure
    """
    # 1. Ensure inputs are tensors
    if not isinstance(atomic_numbers, torch.Tensor):
        atomic_numbers = torch.tensor(atomic_numbers, dtype=torch.long)
    if not isinstance(bond_index, torch.Tensor):
        bond_index = torch.tensor(bond_index, dtype=torch.long)

    # 2. Convert undirected edges to directed edges
    # If the input is already directed, this ensures we have both directions
    edge_index = torch.cat([bond_index, bond_index.flip(0)], dim=1)
    
    # 3. Create edge attributes (bond features)
    num_edges = edge_index.size(1)
    
    if bond_features is not None:
        # Use provided bond features
        if not isinstance(bond_features, torch.Tensor):
            bond_features = torch.tensor(bond_features, dtype=torch.long)
        # Duplicate for bidirectional edges
        edge_attr = torch.cat([bond_features, bond_features], dim=0)
    else:
        # Default: all single bonds (bond type 1)
        edge_attr = torch.ones(num_edges, dtype=torch.long)
    
    # Ensure edge_attr is 2D [num_edges, feature_dim]
    if edge_attr.dim() == 1:
        edge_attr = edge_attr.unsqueeze(1)
    
    # 4. Build the Line Graph (Bond-to-Bond connections)
    # This is the 'Directed' part of DMPNN
    line_graph_edge_index = []
    
    for i in range(num_edges):
        u, v = edge_index[:, i]
        for j in range(num_edges):
            # If edge j starts where edge i ends (v) 
            # AND edge j doesn't go back to where edge i started (u)
            if edge_index[0, j] == v and edge_index[1, j] != u:
                line_graph_edge_index.append([i, j])
                
    line_graph_edge_index = torch.tensor(line_graph_edge_index).t().contiguous() if line_graph_edge_index else torch.empty((2, 0), dtype=torch.long)

    # 5. Ensure x (atomic_numbers) is 2D [num_atoms, feature_dim]
    if atomic_numbers.dim() == 1:
        x = atomic_numbers.unsqueeze(1)
    else:
        x = atomic_numbers

    return Data(
        x=x,                                    # [num_atoms, 1]
        edge_index=edge_index,                  # [2, num_edges]
        edge_attr=edge_attr,                    # [num_edges, 1] - ADDED
        line_graph_edge_index=line_graph_edge_index  # [2, num_line_edges]
    )