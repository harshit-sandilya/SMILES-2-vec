"""
train/data_module.py
===================
LightningDataModule for the 2B-molecule litdata optimized dataset.

Key changes from previous version:
  - setup(stage) properly initialises datasets based on Lightning stage:
      "fit"      -> train + val  (called before trainer.fit())
      "validate" -> val only     (called before trainer.validate())
      "test"     -> test only    (called before trainer.test())
       None      -> all three    (safe fallback)
  - Datasets are created ONCE in setup(), not fresh inside every
    *_dataloader() call (avoids redundant StreamingDataset re-init per epoch)
  - Guards prevent re-initialising a dataset that already exists
    (important when Lightning calls setup() multiple times in one session)
  - Mask ratios are constructor parameters, not hardcoded magic numbers
  - save_hyperparameters() stores all init args for checkpoint reproducibility
  - _build_loader() is a single shared factory -- no duplicated loader logic
"""

from pathlib import Path
from typing import Optional

import pytorch_lightning as pl
from litdata import StreamingDataLoader, StreamingDataset
from torch_geometric.data import Batch

from train.utils import create_masked_graph_from_tensors


class MoleculeDataModule(pl.LightningDataModule):
    """
    Streams litdata chunks from data/optimized/{train,val,test}/ and
    converts stored raw tensors to masked PyG graphs on the fly inside
    DataLoader worker processes.

    Dataset layout expected (written by optimise_dataset.py):
        data/optimized/
            train/   -- ~80% of parquet files (~1.6B molecules)
            val/     -- ~10%  (~200M molecules)
            test/    -- ~10%  (~200M molecules)

    Each stored sample contains:
        atomic_numbers  LongTensor  [MAX_ATOMS]
        bond_matrix     LongTensor  [MAX_ATOMS, MAX_ATOMS]
        smiles          str
        mol_props       FloatTensor [4]  (z-score: logP, MolWt, TPSA, Rings)

    Masking is deferred to collate time so every epoch sees a different
    random mask on the same molecule -- free stochastic augmentation.
    """

    def __init__(
        self,
        data_dir: str = "data/optimized",
        batch_size: int = 256,
        num_workers: int = 32,
        train_mask_atoms: float = 0.25,
        train_mask_bonds: float = 0.25,
        eval_mask_atoms: float = 0.15,
        eval_mask_bonds: float = 0.15,
    ):
        """
        Args:
            data_dir:         Root dir containing train/val/test subdirs.
            batch_size:       Molecules per GPU batch. Scale linearly with
                              GPU count when using DDP.
            num_workers:      DataLoader worker processes. Each worker runs
                              the collate function (RDKit + graph build) in
                              parallel. 32 saturates most HPC nodes.
            train_mask_atoms: Atom mask ratio during training.
            train_mask_bonds: Bond mask ratio during training.
            eval_mask_atoms:  Fixed lower atom ratio for val/test -- gives
                              a stable, comparable loss across checkpoints.
            eval_mask_bonds:  Fixed lower bond ratio for val/test.
        """
        super().__init__()
        self.save_hyperparameters()

        self.data_path = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_mask_atoms = train_mask_atoms
        self.train_mask_bonds = train_mask_bonds
        self.eval_mask_atoms = eval_mask_atoms
        self.eval_mask_bonds = eval_mask_bonds

        # Populated by setup() -- None until then so Lightning can
        # serialise the DataModule without touching the filesystem.
        self.train_dataset: Optional[StreamingDataset] = None
        self.val_dataset: Optional[StreamingDataset] = None
        self.test_dataset: Optional[StreamingDataset] = None

    # ── Setup ────────────────────────────────────────────────────────────

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Instantiate StreamingDatasets based on the Lightning stage string.

        Lightning calls this automatically:
          trainer.fit()      -> setup("fit")      needs train + val
          trainer.validate() -> setup("validate") needs val
          trainer.test()     -> setup("test")     needs test

        The None guards prevent re-initialising a dataset that a previous
        setup() call already created (e.g. calling fit() then test() in
        one script without re-creating the DataModule).
        """
        if stage in ("fit", None):
            if self.train_dataset is None:
                self.train_dataset = StreamingDataset(
                    input_dir=str(self.data_path / "train"),
                    shuffle=True,
                    drop_last=False,
                )
            if self.val_dataset is None:
                self.val_dataset = StreamingDataset(
                    input_dir=str(self.data_path / "val"),
                    shuffle=False,
                    drop_last=False,
                )

        if stage == "validate":
            if self.val_dataset is None:
                self.val_dataset = StreamingDataset(
                    input_dir=str(self.data_path / "val"),
                    shuffle=False,
                    drop_last=False,
                )

        if stage in ("test", None):
            if self.test_dataset is None:
                self.test_dataset = StreamingDataset(
                    input_dir=str(self.data_path / "test"),
                    shuffle=False,
                    drop_last=False,
                )

    # ── Collate functions ────────────────────────────────────────────────

    def collate_graphs(self, batch: list[dict]) -> Batch:
        """
        Training collator -- runs inside DataLoader worker processes.

        Applies stochastic functional-group + random atom/bond masking at
        train_mask_atoms / train_mask_bonds ratios. Every epoch sees a
        different mask for each molecule (free augmentation, zero cost).

        mol_props unsqueeze(0): [4] -> [1, 4] per graph so that
        Batch.from_data_list() stacks to [B, 4] not flattens to [B*4].
        """
        graphs = []
        for item in batch:
            graph = create_masked_graph_from_tensors(
                atomic_numbers=item["atomic_numbers"],
                bond_matrix=item["bond_matrix"],
                mask_ratio_atoms=self.train_mask_atoms,
                mask_ratio_bonds=self.train_mask_bonds,
                smiles=item["smiles"],
                apply_masking=True,
            )
            graph.mol_props = item["mol_props"].unsqueeze(0)  # [1, 4]
            graphs.append(graph)
        return Batch.from_data_list(graphs)

    def collate_graphs_eval(self, batch: list[dict]) -> Batch:
        """
        Val / test collator.

        Uses fixed lower ratios (eval_mask_atoms / eval_mask_bonds) so the
        evaluation loss is stable and directly comparable across checkpoints.
        Still apply_masking=True -- loss measures masked reconstruction
        quality, not trivial identity reconstruction.
        """
        graphs = []
        for item in batch:
            graph = create_masked_graph_from_tensors(
                atomic_numbers=item["atomic_numbers"],
                bond_matrix=item["bond_matrix"],
                mask_ratio_atoms=self.eval_mask_atoms,
                mask_ratio_bonds=self.eval_mask_bonds,
                smiles=item["smiles"],
                apply_masking=True,
            )
            graph.mol_props = item["mol_props"].unsqueeze(0)  # [1, 4]
            graphs.append(graph)
        return Batch.from_data_list(graphs)

    # ── Shared loader factory ────────────────────────────────────────────

    def _build_loader(
        self,
        dataset: StreamingDataset,
        collate_fn,
    ) -> StreamingDataLoader:
        """
        Single factory for all three loaders.

        persistent_workers=True only when num_workers > 0 -- keeps worker
        processes alive between epochs so they don't re-import RDKit and
        re-initialise the tokenizer on every epoch start.
        """
        return StreamingDataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
            persistent_workers=(self.num_workers > 0),
        )

    # ── Dataloaders ──────────────────────────────────────────────────────

    def train_dataloader(self) -> StreamingDataLoader:
        assert self.train_dataset is not None
        return self._build_loader(self.train_dataset, self.collate_graphs)

    def val_dataloader(self) -> StreamingDataLoader:
        assert self.val_dataset is not None
        return self._build_loader(self.val_dataset, self.collate_graphs_eval)

    def test_dataloader(self) -> StreamingDataLoader:
        assert self.test_dataset is not None
        return self._build_loader(self.test_dataset, self.collate_graphs_eval)
