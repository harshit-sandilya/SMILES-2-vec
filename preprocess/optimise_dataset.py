import sys
from pathlib import Path

# =====================================================
# Fix Python path
# =====================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import warnings
from multiprocessing import Pool, cpu_count

import pandas as pd
from litdata import optimize   # <-- IMPORTANT (not lightning.data)

from preprocess.tokenizer import SMILESTokenizer
from train.utils import create_masked_graph_from_tensors, has_max_64_atoms

warnings.filterwarnings(
    "ignore",
    message="An item was larger than the target chunk size",
    category=UserWarning,
)

# ==============================
# Resolve project root & data dir
# ==============================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

tokenizer = None


def init_worker():
    global tokenizer
    tokenizer = SMILESTokenizer()


def process_single_smiles(smiles: str):
    global tokenizer
    try:
        if not has_max_64_atoms(smiles):
            return None

        tokenized_smiles = tokenizer.tokenize(smiles)
        data = create_masked_graph_from_tensors(
            atomic_numbers=tokenized_smiles["atomic_numbers"],
            bond_matrix=tokenized_smiles["bond_matrix"],
            mask_ratio_atoms=0.15,
            mask_ratio_bonds=0.15,
        )
        return data
    except Exception:
        return None


def parallel_process_and_create_graphs(input_file: str):
    csv_chunk_size = 10_000
    pool_chunk_size = 500
    num_processes = cpu_count()

    with Pool(processes=num_processes, initializer=init_worker) as pool:
        for chunk_df in pd.read_csv(input_file, chunksize=csv_chunk_size):
            smiles_list = chunk_df["smiles"].dropna().tolist()
            for data in pool.imap_unordered(
                process_single_smiles,
                smiles_list,
                chunksize=pool_chunk_size,
            ):
                if data is not None:
                    yield data


if __name__ == "__main__":
    input_csv_file = DATA_DIR / "canonical_smiles_subset_100k.csv"
    output_dir = DATA_DIR / "optimized_graph_dataset"

    print(f"Starting dataset optimization for {input_csv_file}...")

    optimize(
        fn=parallel_process_and_create_graphs,
        inputs=[str(input_csv_file)],
        output_dir=str(output_dir),
        num_workers=1,
        chunk_bytes="128MB",
    )

    print("\n✔ Dataset optimization complete!")
    print(f"Streamable dataset created at: {output_dir}")
