import os
import pytorch_lightning as pl
import torch
from lightning.data import StreamingDataset
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class MoleculeDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int = 256, num_workers: int = 4):
        super().__init__()
        # self.data_dir = data_dir
        self.data_dir = os.path.join(PROJECT_ROOT, "data", "optimized_graph_dataset_10k")

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.generator = torch.Generator().manual_seed(42)

    def setup(self, stage: str):
        full_dataset = StreamingDataset(input_dir=self.data_dir)
        self.train_dataset, self.val_dataset = random_split(
            dataset=full_dataset,
            lengths=[0.9, 0.1],
            generator=self.generator,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            # batch_size=self.batch_size,
            batch_size=8,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
        )
