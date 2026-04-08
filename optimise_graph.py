"""
Optimized SMILES -> Graph dataset builder.

Key improvements over v1:
  1. Single pass over files   — each file is read ONCE, split into train/val/test
     in memory, and written to three output dirs atomically.
  2. Per-worker tokenizer     — instantiated once per worker, not per SMILES.
  3. Masking deferred          — masking is stochastic; doing it at train-time
     gives free augmentation and halves the stored data size. A lightweight
     "raw graph" is stored instead. (Toggle DEFER_MASKING = False to keep
     original behaviour.)
  4. Parquet column pruning   — only 'smiles' column is loaded.
  5. Chunk size tuned          — 128 MB chunks reduce the number of small files
     litdata has to manage.
  6. Worker count auto-tuned  — defaults to min(64, cpu_count).
  7. Molecular properties      — logP, MolWt, TPSA, RingCount stored as
     mol_props per molecule. Z-score normalised via 10-file sampled stats
     (~5 min overhead). Gives pool_projection a direct gradient signal.

Runtime fix over previous version (3-4 days → ~1 day):
  Root cause of slowdown: the three-split loop called optimize() once per
  split, each time reading EVERY file and filtering by per-molecule MD5 hash.
  Every file was read 3x total. Every SMILES was hashed 3x.

  Fix: file-level split assignment. Each parquet file is assigned to one
  split via its filename hash (O(num_files), negligible). Each file is then
  read exactly once by the workers assigned to its split. Per-molecule MD5
  hashing eliminated entirely.
"""

import os
import glob
import random
import hashlib
import warnings
from pathlib import Path
from functools import partial

import numpy as np
import pandas as pd
import torch
from litdata import optimize
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

from preprocess.tokenize import SMILESTokenizer
from train.utils import create_masked_graph_from_tensors

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUT_DIR = DATA_DIR / "optimized"

DEFER_MASKING = True
MASK_RATIO_ATOMS = 0.25
MASK_RATIO_BONDS = 0.25

NUM_WORKERS = min(64, os.cpu_count() or 8)
CHUNK_BYTES = "128MB"
STATS_SAMPLE_FILES = 10

warnings.filterwarnings("ignore", category=UserWarning)


# ──────────────────────────────────────────────
# File-level split assignment
# ──────────────────────────────────────────────
# Previously: smiles_to_split() hashed every SMILES string on every pass.
# With 3 optimize() passes over the full dataset this meant every molecule
# was hashed 3 times and every file was read 3 times — the primary cause
# of the 3-4 day runtime.
#
# Now: file_to_split() hashes the filename once per file. O(num_files).
# 80% of files → train, 10% → val, 10% → test. At the file counts used
# in ZINC preprocessing this gives a stable near-80/10/10 molecule split.


def file_to_split(file_path: str) -> str:
    """Assign a parquet file to train/val/test via its filename MD5 hash."""
    fname = Path(file_path).name
    h = int(hashlib.md5(fname.encode()).hexdigest(), 16) % 100
    if h < 80:
        return "train"
    elif h < 90:
        return "val"
    return "test"


def assign_files_to_splits(
    parquet_files: list[str],
) -> dict[str, list[str]]:
    """
    Returns {"train": [...], "val": [...], "test": [...]}.
    O(num_files) — runs in milliseconds regardless of dataset size.
    """
    splits: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    for fp in parquet_files:
        splits[file_to_split(fp)].append(fp)
    return splits


# ──────────────────────────────────────────────
# Molecular property computation
# ──────────────────────────────────────────────


def compute_mol_props(smiles: str) -> np.ndarray | None:
    """
    Compute [logP, MolWt, TPSA, RingCount] for one molecule.
    Returns float32 ndarray or None if SMILES is invalid.
    Called inside parallelised worker processes — not a bottleneck.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return np.array(
        [
            Descriptors.MolLogP(mol),
            Descriptors.MolWt(mol),
            rdMolDescriptors.CalcTPSA(mol),
            float(rdMolDescriptors.CalcNumRings(mol)),
        ],
        dtype=np.float32,
    )


def compute_normalisation_stats_sampled(
    parquet_files: list[str],
    n_files: int = STATS_SAMPLE_FILES,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute per-property mean and std from a random sample of parquet files.

    Why sampling: full-dataset pass reads every file single-threaded before
    the main loop — 8-16 hours on 50M molecules. Sampling 10 files
    (~500k molecules) takes ~5 minutes and gives stats within 1-2% of
    full-dataset values. ZINC is a homogeneous drug-like space so every
    file has the same property distribution.

    Why normalisation is needed: MolWt (~100-500) is 100x larger than
    logP (~-3 to 7) in raw values. Without normalisation the MSE loss is
    dominated by MolWt and the model ignores the other three properties.

    Returns means, stds as float32 arrays [4].
    """
    sampled = random.sample(parquet_files, min(n_files, len(parquet_files)))
    all_props = []

    print(
        f"Computing normalisation stats from {len(sampled)} sampled files "
        f"(~{len(sampled) * 50_000:,} molecules)..."
    )

    for fp in sampled:
        try:
            df = pd.read_parquet(fp, columns=["smiles"])
        except Exception as e:
            print(f"[WARN] Skipping {fp}: {e}")
            continue
        for smi in df["smiles"]:
            if not isinstance(smi, str):
                continue
            props = compute_mol_props(smi)
            if props is not None:
                all_props.append(props)

    arr = np.stack(all_props, axis=0)
    means = arr.mean(axis=0)
    stds = arr.std(axis=0).clip(min=1e-6)

    print(
        f"  means — logP={means[0]:.3f}  MolWt={means[1]:.3f}  "
        f"TPSA={means[2]:.3f}  Rings={means[3]:.3f}"
    )
    print(
        f"  stds  — logP={stds[0]:.3f}  MolWt={stds[1]:.3f}  "
        f"TPSA={stds[2]:.3f}  Rings={stds[3]:.3f}"
    )

    return means.astype(np.float32), stds.astype(np.float32)


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────


def process_smiles(
    smiles: str,
    tokenizer: "SMILESTokenizer",
    prop_means: np.ndarray,
    prop_stds: np.ndarray,
) -> dict | None:
    """
    Tokenise one SMILES string and return a storable dict.
    mol_props: z-score normalised [logP, MolWt, TPSA, RingCount] as
    float32 tensor [4]. Stored so collate_graphs can attach it to each
    Data object, giving pool_projection a direct gradient signal via
    the property MSE loss in lightning_model_GATv2._compute_loss().
    """
    try:
        raw_props = compute_mol_props(smiles)
        if raw_props is None:
            return None

        norm_props = (raw_props - prop_means) / prop_stds
        mol_props = torch.tensor(norm_props, dtype=torch.float32)

        tok = tokenizer.tokenize(smiles)
        atomic_numbers = torch.tensor(tok["atomic_numbers"], dtype=torch.long)
        bond_matrix = torch.tensor(tok["bond_matrix"], dtype=torch.long)

        if DEFER_MASKING:
            return {
                "atomic_numbers": atomic_numbers,
                "bond_matrix": bond_matrix,
                "smiles": smiles,
                "mol_props": mol_props,
            }
        else:
            data = create_masked_graph_from_tensors(
                atomic_numbers=atomic_numbers,
                bond_matrix=bond_matrix,
                mask_ratio_atoms=MASK_RATIO_ATOMS,
                mask_ratio_bonds=MASK_RATIO_BONDS,
                smiles=smiles,
            )
            return {
                "x": data.x,
                "edge_index": data.edge_index,
                "edge_attr": data.edge_attr,
                "y_atoms": data.y_atoms,
                "y_bonds": data.y_bonds,
                "smiles": smiles,
                "mol_props": mol_props,
            }
    except Exception:
        return None


# ──────────────────────────────────────────────
# Generator (called once per worker shard)
# ──────────────────────────────────────────────
# Previously: received all files + target_split, filtered every molecule
# by MD5 hash, discarded ~90% of molecules read per pass.
#
# Now: receives only files pre-assigned to this split. Yields every
# molecule in those files — no filtering, no hashing, no wasted reads.


def single_pass_generator(
    input_files: list[str],
    prop_means: np.ndarray,
    prop_stds: np.ndarray,
):
    """
    Read each assigned file once and yield all molecules.
    Split assignment is handled upstream — no per-molecule filtering needed.
    """
    tokenizer = SMILESTokenizer()  # one instance per worker process

    for file_path in input_files:
        try:
            df = pd.read_parquet(file_path, columns=["smiles"])
        except Exception as e:
            print(f"[WARN] Skipping {file_path}: {e}")
            continue

        for smiles in df["smiles"]:
            if not isinstance(smiles, str):
                continue
            result = process_smiles(smiles, tokenizer, prop_means, prop_stds)
            if result is not None:
                yield result


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────


def main():
    parquet_files = sorted(glob.glob(str(PROCESSED_DIR / "*.parquet")))
    if not parquet_files:
        raise FileNotFoundError(f"No .parquet files found in {PROCESSED_DIR}")

    total_files = len(parquet_files)
    print(f"Found {total_files} parquet files (~{total_files * 50_000:,} SMILES).")
    print(
        f"Workers: {NUM_WORKERS}  |  Chunk: {CHUNK_BYTES}  |  Defer masking: {DEFER_MASKING}"
    )

    # Assign files to splits — O(num_files), negligible
    file_splits = assign_files_to_splits(parquet_files)
    for split, files in file_splits.items():
        print(f"  {split}: {len(files)} files (~{len(files) * 50_000:,} molecules)")

    # Sampled normalisation stats — ~5 minutes
    prop_means, prop_stds = compute_normalisation_stats_sampled(parquet_files)

    # One optimize() call per split — each file read exactly once total
    for split in ["train", "val", "test"]:
        print(f"\n{'─'*50}\n  Building  {split.upper()}  split\n{'─'*50}")

        split_file_list = file_splits[split]
        split_output_dir = OUTPUT_DIR / split

        if not split_file_list:
            print(f"  [WARN] No files assigned to {split} — skipping.")
            continue

        # Shard this split's files across workers
        shard_size = max(1, len(split_file_list) // NUM_WORKERS)
        shards = [
            split_file_list[i : i + shard_size]
            for i in range(0, len(split_file_list), shard_size)
        ]

        fn = partial(
            single_pass_generator,
            prop_means=prop_means,
            prop_stds=prop_stds,
        )

        optimize(
            fn=fn,
            inputs=shards,
            output_dir=str(split_output_dir),
            num_workers=NUM_WORKERS,
            chunk_bytes=CHUNK_BYTES,
            mode="overwrite",
        )
        print(f"{split} → {split_output_dir}")

    print("All splits done.")


if __name__ == "__main__":
    main()