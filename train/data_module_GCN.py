from pathlib import Path

import pandas as pd
import pytorch_lightning as pl
import torch
from torch_geometric.loader import DataLoader

from preprocess.tokenizer import SMILESTokenizer
from train.dataset import MaskedMoleculeDataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class MoleculeDataModuleGCN(pl.LightningDataModule):
    def __init__(
        self,
        data_csv: str,
        batch_size: int = 256,
        num_workers: int = 4,
    ):
        super().__init__()

        self.data_csv = (
            Path(data_csv)
            if Path(data_csv).is_absolute()
            else PROJECT_ROOT / data_csv
        )

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.tokenizer = SMILESTokenizer()

    def setup(self, stage=None):
        # ----------------------------
        # Load CSV (10k subset)
        # ----------------------------
        df = pd.read_csv(self.data_csv)

        smiles_list = df["smiles"].dropna().tolist()

        # ----------------------------
        # Tokenize SMILES
        # ----------------------------
        tokenized_list = []
        for smi in smiles_list:
            try:
                tokenized = self.tokenizer.tokenize(smi)
                tokenized_list.append(tokenized)
            except Exception:
                continue

        # ----------------------------
        # Dataset (NO masking)
        # ----------------------------
        self.dataset = MaskedMoleculeDataset(
            tokenized_list=tokenized_list,
            apply_masking=False,   # 🔑 GCN requirement
        )

    def full_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,  # IMPORTANT for embedding alignment
            num_workers=self.num_workers,
        )
