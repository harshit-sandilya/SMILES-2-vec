import os
import warnings
from multiprocessing import Pool, cpu_count

import pandas as pd
from lightning.data import optimize

from tokenizer import SMILESTokenizer
from utils import create_masked_graph_from_tensors, has_max_64_atoms

warnings.filterwarnings(
    "ignore",
    message="An item was larger than the target chunk size",
    category=UserWarning,
)

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
        with pd.read_csv(input_file, chunksize=csv_chunk_size) as reader:
            for i, chunk_df in enumerate(reader):
                smiles_list = chunk_df["smiles"].dropna().tolist()
                for data in pool.imap_unordered(
                    process_single_smiles, smiles_list, chunksize=pool_chunk_size
                ):
                    if data:
                        yield data


if __name__ == "__main__":
    input_csv_file = "data/canonical_smiles.csv"
    output_dir = "optimized_graph_dataset"

    print(f"Starting dataset optimization for {input_csv_file}...")

    optimize(
        fn=parallel_process_and_create_graphs,
        inputs=[input_csv_file],
        output_dir=output_dir,
        num_workers=1,
        chunk_bytes="128MB",
    )

    print("\nDataset optimization complete!")
    print(f"Your streamable dataset is ready in the '{output_dir}/' directory.")
