from pathlib import Path
import pytorch_lightning as pl
from litdata import StreamingDataset, StreamingDataLoader
from torch_geometric.data import Batch

from train.utils import create_masked_graph_from_tensors


class MoleculeDataModule(pl.LightningDataModule):

    def __init__(
        self,
        data_dir: str = "data/optimized",
        batch_size: int = 256,
        num_workers: int = 32,
    ):
        super().__init__()

        self.data_path = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        pass

    # ---------------- Graph builder ----------------
    def collate_graphs(self, batch):

        graphs = []

        for item in batch:

            graph = create_masked_graph_from_tensors(
                atomic_numbers=item["atomic_numbers"],
                bond_matrix=item["bond_matrix"],
                mask_ratio_atoms=0.25,
                mask_ratio_bonds=0.25,
                smiles=item["smiles"],
                apply_masking=True,
            )
            # unsqueeze(0) stores mol_props as [1, 4] per graph so that
            # Batch.from_data_list() concatenates them to [B, 4] instead
            # of flattening [4] × B into [B*4].
            graph.mol_props = item["mol_props"].unsqueeze(0)  # [1, 4]

            graphs.append(graph)

        return Batch.from_data_list(graphs)

    def collate_graphs_eval(self, batch):
        """
        Separate collator for val/test — masks applied but with
        a fixed lower ratio so evaluation loss is more stable and comparable
        across checkpoints. Using apply_masking=True (not False) ensures the
        val loss is still meaningful as a masked-reconstruction metric.
        """
        graphs = []
        for item in batch:
            graph = create_masked_graph_from_tensors(
                atomic_numbers=item["atomic_numbers"],
                bond_matrix=item["bond_matrix"],
                mask_ratio_atoms=0.15,
                mask_ratio_bonds=0.15,
                apply_masking=True,
                smiles=item["smiles"],
            )
            # unsqueeze(0) stores mol_props as [1, 4] per graph so that
            # Batch.from_data_list() concatenates them to [B, 4] instead
            # of flattening [4] × B into [B*4].
            graph.mol_props = item["mol_props"].unsqueeze(0)  # [1, 4]
            graphs.append(graph)
        return Batch.from_data_list(graphs)

    # ---------------- Train loader ----------------
    def train_dataloader(self):

        dataset = StreamingDataset(
            input_dir=str(self.data_path / "train"),
            shuffle=True,
            drop_last=False,
        )

        return StreamingDataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
            collate_fn=self.collate_graphs,
        )

    # ---------------- Validation loader ----------------
    def val_dataloader(self):

        dataset = StreamingDataset(
            input_dir=str(self.data_path / "val"),
            shuffle=False,
            drop_last=False,
        )

        return StreamingDataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
            collate_fn=self.collate_graphs_eval,
        )

    # ---------------- Test loader ----------------
    def test_dataloader(self):

        dataset = StreamingDataset(
            input_dir=str(self.data_path / "test"),
            shuffle=False,
            drop_last=False,
        )

        return StreamingDataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
            collate_fn=self.collate_graphs_eval,
        )