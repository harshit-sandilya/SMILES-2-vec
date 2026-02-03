import pandas as pd
import torch
from pathlib import Path
import pytorch_lightning as pl
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from preprocess.tokenizer import SMILESTokenizer
from preprocess.dataset import MaskedMoleculeDataset
from train.utils import has_max_64_atoms

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class MoleculeDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 64,
        train_subset_size: int = None,
        num_workers: int = 0,
        mask_ratio_atoms: float = 0.15,     # FIX #4: expose mask ratios
        mask_ratio_bonds: float = 0.15,
    ):
        super().__init__()
        self.data_file = PROJECT_ROOT / data_dir / "canonical_smiles.csv"
        self.batch_size = batch_size
        self.train_subset_size = train_subset_size
        self.num_workers = num_workers
        self.mask_ratio_atoms = mask_ratio_atoms
        self.mask_ratio_bonds = mask_ratio_bonds
        self.generator = torch.Generator().manual_seed(42)

    # --------------------------------------------------
    # FIX #2, #3, #4: use the same tokenizer + dataset
    # that generate_embeddings.py uses, so train and
    # inference see identical graph format.
    # --------------------------------------------------
    def setup(self, stage=None):
        if not self.data_file.exists():
            raise FileNotFoundError(f"Could not find {self.data_file}")

        df = pd.read_csv(self.data_file)
        smiles_list = df["smiles"].tolist()

        if self.train_subset_size:
            smiles_list = smiles_list[: self.train_subset_size]

        # Filter molecules > 64 atoms (same as inference)
        smiles_list = [s for s in smiles_list if has_max_64_atoms(s)]

        # Tokenize with the same tokenizer used at inference time
        tokenizer = SMILESTokenizer()
        tokenized_list = []
        fail = 0

        print(f"📦 Tokenizing {len(smiles_list)} molecules for GIN training")  # FIX #11

        for smi in tqdm(smiles_list):
            try:
                tokenized_list.append(tokenizer.tokenize(smi))
            except Exception:
                fail += 1
                continue

        print(f"✅ Success: {len(tokenized_list)} | ❌ Failed: {fail}")

        # ---------- train / val split on tokenized list ----------
        num_total = len(tokenized_list)
        num_train = int(0.9 * num_total)
        num_val = num_total - num_train

        train_tokens, val_tokens = random_split(
            tokenized_list, [num_train, num_val], generator=self.generator
        )

        # Wrap in MaskedMoleculeDataset — masking is applied per __getitem__
        self.train_dataset = MaskedMoleculeDataset(
            list(train_tokens),
            mask_ratio_atoms=self.mask_ratio_atoms,
            mask_ratio_bonds=self.mask_ratio_bonds,
            apply_masking=True,
        )
        self.val_dataset = MaskedMoleculeDataset(
            list(val_tokens),
            mask_ratio_atoms=self.mask_ratio_atoms,
            mask_ratio_bonds=self.mask_ratio_bonds,
            apply_masking=True,
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
