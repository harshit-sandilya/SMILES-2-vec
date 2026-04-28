#!/usr/bin/env python3
"""
optimise_dataset.py  —  data/processed/ -> data/optimized/ (litdata format)
===========================================================================
Converts canonical, deduplicated parquet shards from build_dataset.py
into litdata binary chunks ready for training.

What this script does NOT do (already guaranteed by build_dataset.py):
  - SMILES validity checks    -- all SMILES are canonical and valid
  - Atom count filtering       -- already capped at MAX_ATOMS in Spark
  - Deduplication              -- done by InChIKey groupBy in Spark
  - SMILES canonicalisation    -- already canonical (RDKit isomericSmiles=True)

What this script does:
  1. File-level train/val/test assignment  (file MD5 hash, O(num_files))
  2. Sample norm stats for mol_props       (~10 files, ~2 min)
  3. SMILES -> raw graph tensors via litdata.optimize()

Key differences from previous version:
  - Column is SMILES (uppercase) -- matches build_dataset.py output schema
  - mol parsed ONCE per molecule; props + tokenizer share the same object
  - tokenize_mol() used instead of tokenize() to avoid double-parse
  - Tokenizer instantiated once per worker, not per molecule
  - PyArrow direct column reads -- faster than pd.read_parquet for one col
  - CHUNK_BYTES = 512MB -- fewer litdata index files, faster training I/O
  - DEFER_MASKING = True always -- masking at collate() = free augmentation
    and halves stored data vs pre-masked graphs
"""

import glob
import hashlib
import os
import random
import warnings
from functools import partial
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from litdata import optimize
from rdkit import RDLogger
from rdkit.Chem import AllChem as AChem
from rdkit.Chem import Descriptors, rdMolDescriptors

from .tokenizer import SMILESTokenizer

RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore", category=UserWarning)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CONFIGURATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TOKENIZER_MAX_ATOMS: int = 64  # Must match MAX_ATOMS in build_dataset.py
DEFER_MASKING: bool = True  # Always True at 2B-mol scale

NUM_WORKERS: int = min(128, os.cpu_count() or 8)
CHUNK_BYTES: str = "512MB"
STATS_NFILES: int = 10  # Files sampled for norm stats (~100k mols)

SMILES_COL = "SMILES"  # Uppercase -- matches build_dataset.py

PROJECT_ROOT = Path(__file__).resolve().parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "data" / "optimized"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SPLIT ASSIGNMENT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _file_to_split(path: str) -> str:
    h = int(hashlib.md5(Path(path).name.encode()).hexdigest(), 16) % 100
    return "train" if h < 80 else ("val" if h < 90 else "test")


def assign_splits(files: list[str]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    for f in files:
        out[_file_to_split(f)].append(f)
    return out


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MOL PROPS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def mol_props_from_mol(mol) -> np.ndarray:
    """
    [logP, MolWt, TPSA, RingCount] as float32 [4].
    Accepts pre-parsed mol -- no internal SMILES re-parse.
    """
    return np.array(
        [
            Descriptors.MolLogP(mol),
            Descriptors.MolWt(mol),
            rdMolDescriptors.CalcTPSA(mol),
            float(rdMolDescriptors.CalcNumRings(mol)),
        ],
        dtype=np.float32,
    )


def compute_norm_stats(
    files: list[str],
    n: int = STATS_NFILES,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample n parquet files and compute per-property mean/std for z-score
    normalisation of mol_props.

    Why sampling: full pass over 201k files is single-threaded here and
    would take hours. 10 files x 10k rows = ~100k mols gives < 2% error
    on mean/std -- the corpus property distribution is stable across files.

    Why z-score: MolWt (100-500) is ~100x larger than logP (-3..7) in raw
    value. Without normalisation MSE loss is dominated by MolWt alone.
    """
    sampled = random.sample(files, min(n, len(files)))
    props: list[np.ndarray] = []

    print(f"[stats] Sampling {len(sampled)} files for normalisation ...")
    for fp in sampled:
        try:
            table = pq.read_table(fp, columns=[SMILES_COL])
        except Exception as e:
            print(f"  [WARN] {fp}: {e}")
            continue
        for smi in table[SMILES_COL].to_pylist():
            if isinstance(smi, str) and smi:
                mol = AChem.MolFromSmiles(smi)
                if mol is not None:
                    props.append(mol_props_from_mol(mol))

    arr = np.stack(props)
    means = arr.mean(0).astype(np.float32)
    stds = arr.std(0).clip(min=1e-6).astype(np.float32)

    labels = ["logP", "MolWt", "TPSA", "Rings"]
    print("  means: " + "  ".join(f"{l}={v:.3f}" for l, v in zip(labels, means)))
    print("  stds:  " + "  ".join(f"{l}={v:.3f}" for l, v in zip(labels, stds)))
    return means, stds


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# WORKER GENERATOR
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def worker_generator(
    input_files: list[str],
    prop_means: np.ndarray,
    prop_stds: np.ndarray,
):
    """
    Called once per shard by litdata.optimize().

    Mol parse strategy — single AChem.MolFromSmiles call per molecule:
      mol  ->  mol_props_from_mol(mol)      [no re-parse, direct mol use]
           ->  tokenizer.tokenize_mol(mol)  [no re-parse, new method]

    No filtering -- build_dataset.py already guaranteed:
      - Valid canonical SMILES
      - 3 <= heavy atoms <= TOKENIZER_MAX_ATOMS
      - No duplicates (InChIKey-deduplicated across all sources)

    Exceptions silenced: at 2B mol scale even 0.0001% failure = 2000 lines.
    The Spark pre-processing safety net makes these essentially impossible.

    Stored per molecule (DEFER_MASKING=True):
      atomic_numbers  LongTensor [MAX_ATOMS]            zero-padded
      bond_matrix     LongTensor [MAX_ATOMS, MAX_ATOMS]  symmetric
      smiles          str   needed by collate_graphs -> create_masked_graph
      mol_props       FloatTensor [4]                   z-score normalised
    """
    tokenizer = SMILESTokenizer(max_atoms=TOKENIZER_MAX_ATOMS)

    for file_path in input_files:
        try:
            table = pq.read_table(file_path, columns=[SMILES_COL])
        except Exception as e:
            print(f"[WARN] Cannot read {file_path}: {e}")
            continue

        for smiles in table[SMILES_COL].to_pylist():
            if not isinstance(smiles, str) or not smiles:
                continue
            try:
                # Single parse -- mol shared by props AND tokenizer
                mol = AChem.MolFromSmiles(smiles)
                if mol is None:  # safety net; should never trigger
                    continue

                raw_props = mol_props_from_mol(mol)
                norm_props = (raw_props - prop_means) / prop_stds
                mol_props = torch.tensor(norm_props, dtype=torch.float32)

                # tokenize_mol avoids a second internal Chem.MolFromSmiles
                tok = tokenizer.tokenize_mol(mol, smiles=smiles)

                yield {
                    "atomic_numbers": tok["atomic_numbers"],  # LongTensor [MAX_ATOMS]
                    "bond_matrix": tok[
                        "bond_matrix"
                    ],  # LongTensor [MAX_ATOMS, MAX_ATOMS]
                    "smiles": smiles,  # str
                    "mol_props": mol_props,  # FloatTensor [4]
                }

            except Exception:
                continue


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def main():
    parquet_files = sorted(glob.glob(str(PROCESSED_DIR / "*.parquet")))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {PROCESSED_DIR}")

    n = len(parquet_files)
    print(f"Found {n:,} parquet files  (~{n * 10_000:,} molecules)")
    print(
        f"Workers={NUM_WORKERS}  |  Chunk={CHUNK_BYTES}  |  DeferMask={DEFER_MASKING}"
    )
    print()

    splits = assign_splits(parquet_files)
    for split, fs in splits.items():
        print(f"  {split:5s}: {len(fs):7,} files  (~{len(fs) * 10_000:,} mols)")
    print()

    prop_means, prop_stds = compute_norm_stats(parquet_files)

    for split in ("train", "val", "test"):
        split_files = splits[split]
        if not split_files:
            print(f"  [WARN] No files assigned to {split} -- skipping.")
            continue

        out_dir = OUTPUT_DIR / split
        print(f"\n{'━' * 56}")
        print(
            f"  {split.upper()}  |  {len(split_files):,} files  "
            f"|  ~{len(split_files) * 10_000:,} molecules"
        )
        print(f"{'━' * 56}")

        shard_sz = max(1, len(split_files) // NUM_WORKERS)
        shards = [
            split_files[i : i + shard_sz] for i in range(0, len(split_files), shard_sz)
        ]

        optimize(
            fn=partial(
                worker_generator,
                prop_means=prop_means,
                prop_stds=prop_stds,
            ),
            inputs=shards,
            output_dir=str(out_dir),
            num_workers=NUM_WORKERS,
            chunk_bytes=CHUNK_BYTES,
            mode="overwrite",
        )
        print(f"  -> {out_dir}")

    print("\nAll splits done.")


if __name__ == "__main__":
    main()
