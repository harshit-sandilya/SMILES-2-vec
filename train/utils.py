import torch
from rdkit.Chem import AllChem as Chem
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from config import MASK_ATOM_ID, MASK_BOND_ID


def has_max_64_atoms(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    return mol.GetNumAtoms() <= 64


def create_graph_data(
    tokenized_item,
    mask_ratio_atoms=0.15,
    mask_ratio_bonds=0.15,
):
    num_atoms = tokenized_item["num_atoms"].item()
    atomic_feats = tokenized_item["atomic_numbers"][:num_atoms].clone()
    atom_labels = torch.full_like(atomic_feats, -1)

    num_atoms_to_mask = int(num_atoms * mask_ratio_atoms)
    if num_atoms_to_mask > 0:
        mask_indices = torch.randperm(num_atoms)[:num_atoms_to_mask]
        atom_labels[mask_indices] = atomic_feats[mask_indices]
        atomic_feats[mask_indices] = MASK_ATOM_ID          # FIX #12: use imported constant

    bond_matrix = tokenized_item["bond_matrix"][:num_atoms, :num_atoms]
    edge_index = bond_matrix.nonzero().t().contiguous()
    edge_attr = bond_matrix[edge_index[0], edge_index[1]].clone()

    # --------------------------------------------------
    # FIX #5: symmetric bond masking.
    # Pick unique bonds (i < j), mask them, then apply the
    # same mask to their reverse edges (j, i) so the model
    # cannot peek at the unmasked mirror edge.
    # --------------------------------------------------
    bond_labels = torch.full_like(edge_attr, -1)

    unique_edge_mask = edge_index[0] < edge_index[1]
    unique_edge_indices = torch.where(unique_edge_mask)[0]
    num_unique_bonds = len(unique_edge_indices)
    num_bonds_to_mask = int(num_unique_bonds * mask_ratio_bonds)

    if num_bonds_to_mask > 0:
        mask_bond_perm = torch.randperm(num_unique_bonds)[:num_bonds_to_mask]
        forward_indices = unique_edge_indices[mask_bond_perm]

        # Save labels & mask forward edges
        bond_labels[forward_indices] = edge_attr[forward_indices]
        edge_attr[forward_indices] = MASK_BOND_ID

        # Find and mask the corresponding reverse edges (j -> i)
        fwd_src = edge_index[0][forward_indices]   # original i
        fwd_dst = edge_index[1][forward_indices]   # original j
        # Reverse edge has src=j, dst=i
        rev_mask = (edge_index[0].unsqueeze(1) == fwd_dst.unsqueeze(0)) & \
                   (edge_index[1].unsqueeze(1) == fwd_src.unsqueeze(0))
        rev_indices = torch.where(rev_mask.any(dim=1))[0]

        bond_labels[rev_indices] = edge_attr[rev_indices]
        edge_attr[rev_indices] = MASK_BOND_ID

    graph_data = Data(
        x=atomic_feats.unsqueeze(-1),           # consistent shape [N, 1]
        edge_index=edge_index,
        edge_attr=edge_attr,
        y_atoms=atom_labels,
        y_bonds=bond_labels,
        num_atoms=num_atoms,
        smile=tokenized_item["smile"],
    )
    return graph_data


def create_masked_graph_from_tensors(
    atomic_numbers: torch.Tensor,
    bond_matrix: torch.Tensor,
    mask_ratio_atoms: float,
    mask_ratio_bonds: float,
    apply_masking: bool = True,
) -> Data:
    num_atoms = (atomic_numbers != 0).sum().item()

    # -----------------------------
    # Node features
    # -----------------------------
    atomic_feats = atomic_numbers[:num_atoms].clone()

    # -----------------------------
    # Build edges
    # -----------------------------
    bond_matrix_unpadded = bond_matrix[:num_atoms, :num_atoms]
    edge_index = bond_matrix_unpadded.nonzero().t().contiguous()
    edge_attr = bond_matrix_unpadded[edge_index[0], edge_index[1]].clone()

    # -----------------------------
    # Optional masking
    # -----------------------------
    if apply_masking:
        # --- Atom masking ---
        atom_labels = torch.full_like(atomic_feats, -1)
        num_atoms_to_mask = int(num_atoms * mask_ratio_atoms)

        if num_atoms_to_mask > 0:
            mask_indices = torch.randperm(num_atoms)[:num_atoms_to_mask]
            atom_labels[mask_indices] = atomic_feats[mask_indices]
            atomic_feats[mask_indices] = MASK_ATOM_ID

        # --- Bond masking (symmetric) ---                # FIX #5
        bond_labels = torch.full_like(edge_attr, -1)
        unique_edge_mask = edge_index[0] < edge_index[1]
        unique_edge_indices = torch.where(unique_edge_mask)[0]
        num_unique_bonds = len(unique_edge_indices)
        num_bonds_to_mask = int(num_unique_bonds * mask_ratio_bonds)

        if num_bonds_to_mask > 0:
            mask_bond_perm = torch.randperm(num_unique_bonds)[:num_bonds_to_mask]
            forward_indices = unique_edge_indices[mask_bond_perm]

            # Mask forward edges
            bond_labels[forward_indices] = edge_attr[forward_indices]
            edge_attr[forward_indices] = MASK_BOND_ID

            # Mask reverse edges
            fwd_src = edge_index[0][forward_indices]
            fwd_dst = edge_index[1][forward_indices]
            rev_mask = (edge_index[0].unsqueeze(1) == fwd_dst.unsqueeze(0)) & \
                       (edge_index[1].unsqueeze(1) == fwd_src.unsqueeze(0))
            rev_indices = torch.where(rev_mask.any(dim=1))[0]

            bond_labels[rev_indices] = edge_attr[rev_indices]
            edge_attr[rev_indices] = MASK_BOND_ID

        return Data(
            x=atomic_feats.unsqueeze(-1),
            edge_index=edge_index,
            edge_attr=edge_attr,
            y_atoms=atom_labels,
            y_bonds=bond_labels,
        )

    # -----------------------------
    # Inference path (NO masking)
    # -----------------------------
    return Data(
        x=atomic_feats.unsqueeze(-1),
        edge_index=edge_index,
        edge_attr=edge_attr,
    )


def get_single_embedding(smiles_string, model, tokenizer, device):
    try:
        tokenized = tokenizer.tokenize(smiles_string)
        data = create_masked_graph_from_tensors(
            tokenized["atomic_numbers"],
            tokenized["bond_matrix"],
            mask_ratio_atoms=0,
            mask_ratio_bonds=0,
            apply_masking=False,
        )
        loader = DataLoader([data], batch_size=1, shuffle=False)
        batch = next(iter(loader)).to(device)
        with torch.no_grad():
            embedding = model.get_embedding(batch)
        return embedding.cpu().numpy()
    except Exception as e:
        print(f"Error processing SMILES '{smiles_string}': {e}")
        return None


def create_graph_data_for_inference(tokenized_item):
    if tokenized_item is None:
        return None
    num_atoms = tokenized_item["num_atoms"].item()
    atomic_feats = tokenized_item["atomic_numbers"][:num_atoms]
    bond_matrix = tokenized_item["bond_matrix"][:num_atoms, :num_atoms]
    edge_index = bond_matrix.nonzero().t().contiguous()
    edge_attr = bond_matrix[edge_index[0], edge_index[1]]
    return Data(
        x=atomic_feats.unsqueeze(-1),          # consistent shape [N, 1]
        edge_index=edge_index,
        edge_attr=edge_attr,
        smile=tokenized_item["smile"],
    )
