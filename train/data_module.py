from pathlib import Path

import pytorch_lightning as pl
import torch
from lightning.data import StreamingDataset
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader


# ---------------- Project root ----------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent


class MoleculeDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int = 256, num_workers: int = 4):
        super().__init__()

        self.data_dir = (
            Path(data_dir)
            if Path(data_dir).is_absolute()
            else PROJECT_ROOT / data_dir
        )
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.generator = torch.Generator().manual_seed(42)

    def setup(self, stage: str | None = None):
        full_dataset = StreamingDataset(input_dir=str(self.data_dir))

        self.train_dataset, self.val_dataset = random_split(
            dataset=full_dataset,
            lengths=[0.9, 0.1],
            generator=self.generator,
        )

    def full_dataloader(self):
        full_dataset = StreamingDataset(input_dir=str(self.data_dir))
        return DataLoader(
        full_dataset,
        batch_size=self.batch_size,
        num_workers=self.num_workers,
        shuffle=False,
    )


    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

