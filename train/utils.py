import torch
from rdkit import Chem as RChem
from rdkit.Chem import AllChem as Chem
from rdkit.Chem import MolFromSmarts
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from train.config import MASK_BOND_ID

# ==============================================================
# Rich atom feature extractor
# ==============================================================
# Previously every atom was encoded as a single integer (atomic number).
# Carbon in benzene, a carbonyl, and a methyl group all mapped to 6 —
# the model had to infer chemical context purely from graph topology.
#
# This function extracts 7 chemical properties per atom directly from
# RDKit so the model receives genuine chemical signal at the input layer.
# Each property is stored as an integer and embedded separately in the
# model via dedicated nn.Embedding tables (see model_GATv2.py).
#
# Column layout of the returned [N, 7] tensor:
#   0: atomic_number          — element identity (1–118, or MASK_ATOM_ID)
#   1: degree                 — number of bonds (0–10, clamped)
#   2: formal_charge + 4      — ionic state, offset so –4..+4 → 0..8
#   3: total_num_Hs           — implicit + explicit H count (0–8, clamped)
#   4: is_aromatic            — 0 or 1
#   5: is_in_ring             — 0 or 1
#   6: hybridization          — SP=0, SP2=1, SP3=2, SP3D=3, SP3D2=4, other=5

HYBRIDIZATION_MAP = {
    RChem.rdchem.HybridizationType.SP: 0,
    RChem.rdchem.HybridizationType.SP2: 1,
    RChem.rdchem.HybridizationType.SP3: 2,
    RChem.rdchem.HybridizationType.SP3D: 3,
    RChem.rdchem.HybridizationType.SP3D2: 4,
}


def get_atom_features(mol) -> torch.Tensor:
    """
    Returns a [N, 7] long tensor of chemical atom features for all
    atoms in `mol`. Used by create_masked_graph_from_tensors when a
    valid RDKit mol object is available.
    """
    feats = []
    for atom in mol.GetAtoms():
        feats.append(
            [
                atom.GetAtomicNum(),
                min(atom.GetDegree(), 10),
                atom.GetFormalCharge() + 4,  # shift –4..+4 → 0..8
                min(atom.GetTotalNumHs(), 8),
                int(atom.GetIsAromatic()),
                int(atom.IsInRing()),
                HYBRIDIZATION_MAP.get(atom.GetHybridization(), 5),
            ]
        )
    return torch.tensor(feats, dtype=torch.long)  # [N, 7]


# ==============================================================
# Bond feature extractor
# ==============================================================
# Column layout of the returned [num_bonds, 5] tensor:
#   0: bond_type       — 1=single, 2=double, 3=triple, 4=aromatic
#   1: is_conjugated   — 0 or 1
#   2: is_in_ring      — 0 or 1
#   3: ring_size       — smallest ring containing bond (0 if none, capped 8)
#   4: stereo          — 0=none, 1=any, 2=E, 3=Z
#
# One row per RDKit bond (undirected). Directed-edge mapping is done
# separately in get_edge_bond_features().

BOND_TYPE_MAP = {
    RChem.rdchem.BondType.SINGLE: 1,
    RChem.rdchem.BondType.DOUBLE: 2,
    RChem.rdchem.BondType.TRIPLE: 3,
    RChem.rdchem.BondType.AROMATIC: 4,
}

STEREO_MAP = {
    RChem.rdchem.BondStereo.STEREONONE: 0,
    RChem.rdchem.BondStereo.STEREOANY: 1,
    RChem.rdchem.BondStereo.STEREOE: 2,
    RChem.rdchem.BondStereo.STEREOZ: 3,
}


def get_bond_features(mol) -> torch.Tensor:
    """Returns a [num_bonds, 5] long tensor. One row per RDKit bond."""
    ring_info = mol.GetRingInfo()
    feats = []

    for bond in mol.GetBonds():
        bond_rings = [r for r in ring_info.BondRings() if bond.GetIdx() in r]
        min_ring = min(len(r) for r in bond_rings) if bond_rings else 0
        min_ring = min(min_ring, 8)

        feats.append(
            [
                BOND_TYPE_MAP.get(bond.GetBondType(), 1),
                int(bond.GetIsConjugated()),
                int(bond.IsInRing()),
                min_ring,
                STEREO_MAP.get(bond.GetStereo(), 0),
            ]
        )

    if not feats:
        return torch.zeros((0, 5), dtype=torch.long)
    return torch.tensor(feats, dtype=torch.long)  # [num_bonds, 5]


def get_edge_bond_features(mol, edge_index: torch.Tensor) -> torch.Tensor:
    """
    Maps undirected bond features to every directed edge in edge_index.
    Returns [E, 5] long tensor.

    Uses a dict keyed by (src, dst) — O(num_bonds) build, O(1) lookup.
    """
    bond_feats = get_bond_features(mol)
    num_edges = edge_index.size(1)
    edge_bond_feats = torch.zeros(num_edges, 5, dtype=torch.long)

    if bond_feats.size(0) == 0:
        return edge_bond_feats

    bond_lookup: dict[tuple[int, int], int] = {}
    for idx, bond in enumerate(mol.GetBonds()):
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bond_lookup[(i, j)] = idx
        bond_lookup[(j, i)] = idx

    src = edge_index[0].tolist()
    dst = edge_index[1].tolist()
    for edge_idx, (i, j) in enumerate(zip(src, dst)):
        bond_idx = bond_lookup.get((i, j))
        if bond_idx is not None:
            edge_bond_feats[edge_idx] = bond_feats[bond_idx]

    return edge_bond_feats  # [E, 5]


# ==============================================================
# Functional group SMARTS patterns for harder masking
# ==============================================================
# Random 15% atom masking is too easy on ZINC because ~50–60% of atoms
# are Carbon — the model learns to predict Carbon by default and achieves
# low loss without understanding chemistry.
#
# Masking entire functional groups forces the model to use broad molecular
# context (the surrounding scaffold) to reconstruct the masked region,
# rather than relying on immediate neighbours.
#
# Strategy:
#   1. Find all functional group matches in the molecule via SMARTS.
#   2. Randomly decide whether to mask each match (50% chance per match).
#   3. Top-up with random atom masking to reach the target mask ratio.
#   4. All atoms in fg_masked have ALL 7 feature columns zeroed out.

FG_SMARTS = [
    "[CX3]=[OX1]",  # carbonyl  (C=O)
    "[NX3][CX3]=[OX1]",  # amide     (NC=O)
    "[OX2H]",  # hydroxyl  (–OH)
    "[#6][F,Cl,Br,I]",  # halide    (C–X)
    "[NX3;H2,H1;!$(NC=O)]",  # amine     (–NH2 / –NH–)
    "c1ccccc1",  # benzene ring
    "[CX3](=O)[OX2H1]",  # carboxylic acid (–COOH)
    "[#6][SX2H]",  # thiol     (–SH)
    "[NX2]=[CX3]",  # imine     (C=N)
]
FG_QUERIES = [MolFromSmarts(s) for s in FG_SMARTS]


def has_max_64_atoms(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    return mol.GetNumAtoms() <= 64


def create_masked_graph_from_tensors(
    atomic_numbers: torch.Tensor,
    bond_matrix: torch.Tensor,
    mask_ratio_atoms: float,
    mask_ratio_bonds: float,
    smiles: str,
    apply_masking: bool = True,
) -> Data:
    mol = RChem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit cannot parse SMILES: {smiles!r}")

    num_atoms = mol.GetNumAtoms()

    atomic_feats = get_atom_features(mol)  # [num_atoms, 7]
    assert atomic_feats.shape[0] == num_atoms

    bond_matrix_unpadded = bond_matrix[:num_atoms, :num_atoms]
    edge_index = bond_matrix_unpadded.nonzero().t().contiguous()

    # Validate before any GPU touch
    if edge_index.numel() > 0:
        assert edge_index.max() < num_atoms, (
            f"edge_index OOB: max={edge_index.max()}, num_atoms={num_atoms}, smiles={smiles!r}"
        )

    edge_attr = bond_matrix_unpadded[edge_index[0], edge_index[1]].clone()
    edge_bond_feats = get_edge_bond_features(mol, edge_index)

    # -----------------------------
    # Optional masking
    # -----------------------------
    if apply_masking:
        # --- Atom masking ---
        atom_labels = torch.full((num_atoms,), -1, dtype=torch.long)
        bond_labels = torch.full((edge_attr.size(0),), -1, dtype=torch.long)

        # --------------------------------------------------------------
        # Functional group masking
        # Identifies atoms belonging to chemical functional groups and
        # masks them as a unit, forcing the model to reason about the
        # broader molecular scaffold rather than local neighbourhood.
        # --------------------------------------------------------------
        fg_masked = set()
        for query in FG_QUERIES:
            for match in mol.GetSubstructMatches(query):
                if torch.rand(1).item() < 0.5:
                    fg_masked.update(match)

        # Top-up with random masking to reach target ratio
        target_n_masked = max(int(num_atoms * mask_ratio_atoms), 1)
        unmasked = [i for i in range(num_atoms) if i not in fg_masked]
        extra_needed = max(0, target_n_masked - len(fg_masked))
        if extra_needed > 0 and unmasked:
            non_carbon = [i for i in unmasked if atomic_feats[i, 0] != 6]
            carbon = [i for i in unmasked if atomic_feats[i, 0] == 6]
            if non_carbon:
                nc_perm = torch.randperm(len(non_carbon)).tolist()
                non_carbon = [non_carbon[p] for p in nc_perm]
            if carbon:
                c_perm = torch.randperm(len(carbon)).tolist()
                carbon = [carbon[p] for p in c_perm]
            priority_pool = non_carbon + carbon
            fg_masked.update(priority_pool[:extra_needed])

        # Apply atom masking
        if fg_masked:
            mask_indices = torch.tensor(sorted(fg_masked), dtype=torch.long)
            # Label = atomic number from column 0 (what the model predicts)
            atom_labels[mask_indices] = atomic_feats[mask_indices, 0]
            # Zero out ALL feature columns for masked atoms so the model
            # cannot peek at degree, aromaticity, etc. of the masked atom
            atomic_feats[mask_indices] = 0

        # --------------------------------------------------------------
        # Symmetric bond masking (logic unchanged from original)
        # Picks unique bonds (i < j), masks them and their reverse edges
        # so the model cannot peek at the unmasked mirror edge.
        # --------------------------------------------------------------
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
            edge_bond_feats[forward_indices] = 0

            # Mask reverse edges
            fwd_src = edge_index[0][forward_indices]
            fwd_dst = edge_index[1][forward_indices]
            rev_mask = (edge_index[0].unsqueeze(1) == fwd_dst.unsqueeze(0)) & (
                edge_index[1].unsqueeze(1) == fwd_src.unsqueeze(0)
            )
            rev_indices = torch.where(rev_mask.any(dim=1))[0]

            bond_labels[rev_indices] = edge_attr[rev_indices]
            edge_attr[rev_indices] = MASK_BOND_ID
            edge_bond_feats[rev_indices] = 0

        return Data(
            x=atomic_feats,
            edge_index=edge_index,
            edge_attr=edge_attr,
            edge_bond_feats=edge_bond_feats,
            y_atoms=atom_labels,
            y_bonds=bond_labels,
        )

    # -----------------------------
    # Inference path (NO masking)
    # -----------------------------
    return Data(
        x=atomic_feats,
        edge_index=edge_index,
        edge_attr=edge_attr,
        edge_bond_feats=edge_bond_feats,
    )


def get_single_embedding(smiles_string, model, tokenizer, device):
    try:
        tokenized = tokenizer.tokenize(smiles_string)
        data = create_masked_graph_from_tensors(
            atomic_numbers=tokenized["atomic_numbers"],
            bond_matrix=tokenized["bond_matrix"],
            mask_ratio_atoms=0,
            mask_ratio_bonds=0,
            smiles=smiles_string,
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


def get_batch_embeddings(smiles_list, model, tokenizer, device, batch_size=256):
    """
    Vectorised batch inference — avoids creating a new DataLoader per molecule.
    Use this for embedding extraction after training.
    """
    graphs = []
    for smi in smiles_list:
        try:
            tokenized = tokenizer.tokenize(smi)
            graph = create_masked_graph_from_tensors(
                atomic_numbers=tokenized["atomic_numbers"],
                bond_matrix=tokenized["bond_matrix"],
                mask_ratio_atoms=0,
                mask_ratio_bonds=0,
                smiles=smi,
                apply_masking=False,
            )
            graphs.append(graph)
        except Exception as e:
            print(f"Skipping '{smi}': {e}")

    loader = DataLoader(graphs, batch_size=batch_size, shuffle=False)
    embeddings = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            embeddings.append(model.get_embedding(batch.to(device)).cpu())
    return torch.cat(embeddings).numpy()
