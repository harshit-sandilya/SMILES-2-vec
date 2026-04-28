"""
preprocess/tokenizer.py
======================
SMILESTokenizer: SMILES string -> atomic_numbers + bond_matrix tensors.
"""

import numpy as np
import torch
from rdkit.Chem import AllChem as Chem


class SMILESTokenizer:
    BOND_TYPE_MAP = {
        Chem.rdchem.BondType.SINGLE: 1,
        Chem.rdchem.BondType.DOUBLE: 2,
        Chem.rdchem.BondType.TRIPLE: 3,
        Chem.rdchem.BondType.AROMATIC: 4,
    }

    def __init__(self, max_atoms: int = 64):
        self.max_atoms = max_atoms

    # ── Public API ──────────────────────────────────────────────────────

    def tokenize(self, smiles: str) -> dict:
        """SMILES string -> graph tensors. Raises ValueError on bad SMILES."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles!r}")
        return self._mol_to_dict(mol, smiles)

    def tokenize_mol(self, mol, smiles: str = "") -> dict:
        """
        Pre-parsed RDKit mol -> graph tensors.

        Use this when the caller already holds a mol object to avoid a
        second Chem.MolFromSmiles call. optimise_dataset.py uses this so
        that mol_props and tokenisation share a single mol parse.

        Args:
            mol:    Valid rdkit.Chem.Mol. Raises ValueError if None.
            smiles: Original SMILES string, stored under "smile" key.
        """
        if mol is None:
            raise ValueError("tokenize_mol received None mol.")
        return self._mol_to_dict(mol, smiles)

    # ── Internal ────────────────────────────────────────────────────────

    def _mol_to_dict(self, mol, smiles: str) -> dict:
        num_atoms = mol.GetNumAtoms()
        if num_atoms > self.max_atoms:
            raise ValueError(
                f"Molecule has {num_atoms} atoms > max_atoms={self.max_atoms}. "
                f"SMILES: {smiles!r}"
            )

        atomic_numbers = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
        atomic_numbers += [0] * (self.max_atoms - num_atoms)  # zero-pad

        bond_matrix = np.zeros((self.max_atoms, self.max_atoms), dtype=np.int64)
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            btype = self.BOND_TYPE_MAP.get(bond.GetBondType(), 0)
            bond_matrix[i, j] = btype
            bond_matrix[j, i] = btype

        return {
            "smile": smiles,
            "num_atoms": torch.tensor(num_atoms, dtype=torch.long),
            "atomic_numbers": torch.tensor(atomic_numbers, dtype=torch.long),
            "bond_matrix": torch.tensor(bond_matrix, dtype=torch.long),
        }
