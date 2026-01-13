import numpy as np
import torch
from rdkit.Chem import AllChem as Chem


class SMILESTokenizer:
    def __init__(self, max_atoms=64):
        self.max_atoms = max_atoms
        self.bond_type_map = {
            Chem.rdchem.BondType.SINGLE: 1,
            Chem.rdchem.BondType.DOUBLE: 2,
            Chem.rdchem.BondType.TRIPLE: 3,
            Chem.rdchem.BondType.AROMATIC: 4,
        }

    def tokenize(self, smiles: str):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")

        num_atoms = mol.GetNumAtoms()
        if num_atoms > self.max_atoms:
            raise ValueError(f"Too many atoms: {num_atoms} > {self.max_atoms}")

        atomic_numbers = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
        atomic_numbers += [0] * (self.max_atoms - num_atoms)

        bond_matrix = np.zeros((self.max_atoms, self.max_atoms), dtype=int)
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            btype = self.bond_type_map.get(bond.GetBondType(), 0)
            bond_matrix[i, j] = btype
            bond_matrix[j, i] = btype

        return {
            "smile": smiles,
            "num_atoms": torch.tensor(num_atoms, dtype=torch.long),
            "atomic_numbers": torch.tensor(atomic_numbers, dtype=torch.long),
            "bond_matrix": torch.tensor(bond_matrix, dtype=torch.long),
        }
