import pandas as pd
import torch
from pathlib import Path
import pytorch_lightning as pl
from torch.utils.data import random_split
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from rdkit import Chem
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class MoleculeDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 64,
        train_subset_size: int = None,
        num_workers: int = 0,
        mask_ratio_atoms: float = 0.15,  # Add this
        mask_ratio_bonds: float = 0.15,  # Add this
    ):
        super().__init__()
        self.data_file = PROJECT_ROOT / data_dir / "canonical_smiles.csv"
        self.batch_size = batch_size
        self.train_subset_size = train_subset_size
        self.num_workers = num_workers
        self.mask_ratio_atoms = mask_ratio_atoms  # Store it
        self.mask_ratio_bonds = mask_ratio_bonds  # Store it
        self.generator = torch.Generator().manual_seed(42)

    def smiles_to_graph(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        # Node features: atomic numbers
        x = torch.tensor(
            [[atom.GetAtomicNum()] for atom in mol.GetAtoms()],
            dtype=torch.float
        )

        # Edges
        edge_index = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_index.append([i, j])
            edge_index.append([j, i])  # undirected

        if len(edge_index) == 0:
            return None

        edge_index = torch.tensor(edge_index, dtype=torch.long).t()

        return Data(x=x, edge_index=edge_index)

    def setup(self, stage=None):
        if not self.data_file.exists():
            raise FileNotFoundError(f"Could not find {self.data_file}")

        df = pd.read_csv(self.data_file)
        smiles_list = df["smiles"].tolist()

        if self.train_subset_size:
            smiles_list = smiles_list[:self.train_subset_size]

        dataset = []
        fail = 0

        print(f"📦 Building GATv2 graphs from {len(smiles_list)} molecules")

        for smi in tqdm(smiles_list):
            data = self.smiles_to_graph(smi)
            if data is None:
                fail += 1
                continue
            dataset.append(data)

        print(f"✅ Success: {len(dataset)} | ❌ Failed: {fail}")

        num_total = len(dataset)
        num_train = int(0.9 * num_total)
        num_val = num_total - num_train

        self.train_dataset, self.val_dataset = random_split(
            dataset, [num_train, num_val], generator=self.generator
        )

        print(f"Train: {len(self.train_dataset)} | Val: {len(self.val_dataset)}")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )