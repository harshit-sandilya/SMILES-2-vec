import argparse
import os
from functools import partial
from multiprocessing import Pool, cpu_count

import pandas as pd
from rdkit import Chem
from tqdm import tqdm


def canonicalize_smiles(smiles: str, isomeric: bool) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=isomeric)
    return None


def process_file_parallel(
    input_path: str,
    output_path: str,
    keep_isomers: bool,
    chunksize: int,
    num_workers: int,
):
    try:
        total_rows = sum(1 for _ in open(input_path)) - 1
        if total_rows <= 0:
            print(f"Error: The file {input_path} is empty or has only a header.")
            return
    except FileNotFoundError:
        print(f"Error: The input file was not found at {input_path}")
        return

    print(f"Starting processing of {total_rows} molecules from {input_path}.")
    print(f"Using {num_workers} worker processes.")

    reader = pd.read_csv(input_path, chunksize=chunksize)
    worker_func = partial(canonicalize_smiles, isomeric=keep_isomers)
    header_written = False
    processed_rows = 0

    with Pool(processes=num_workers) as pool:
        with tqdm(total=total_rows, desc="Canonicalizing SMILES", unit="mol") as pbar:
            for chunk in reader:
                if "smiles" not in chunk.columns:
                    raise ValueError("Input CSV must have a 'smiles' column.")

                canonical_results = pool.map(worker_func, chunk["smiles"])
                chunk["smiles"] = canonical_results
                chunk.dropna(subset=["smiles"], inplace=True)

                if not chunk.empty:
                    if not header_written:
                        chunk.to_csv(output_path, index=False, mode="w")
                        header_written = True
                    else:
                        chunk.to_csv(output_path, index=False, mode="a", header=False)

                pbar.update(len(chunk))
                processed_rows += len(chunk)

    print("-" * 30)
    print(f"[✔] Processing complete.")
    print(f"Total valid molecules processed: {processed_rows}")
    print(f"Canonical SMILES saved to {output_path}")
    print("-" * 30)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Efficiently canonicalize SMILES in a large CSV file using parallel processing.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--input",
        default="data/smiles.csv",
        type=str,
        help="Path to the input CSV file (must contain a 'smiles' column).",
    )
    parser.add_argument(
        "--output",
        default="data/canonical_smiles.csv",
        type=str,
        help="Path to save the processed output CSV file.",
    )
    parser.add_argument(
        "--keep-isomers",
        action="store_true",
        help="Include stereochemistry and isotope information in the canonical SMILES.",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=5000,
        help="Number of SMILES to process in each chunk (default: 5000).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of CPU cores to use for processing (default: all available cores).",
    )
    args = parser.parse_args()

    num_workers = args.num_workers if args.num_workers else cpu_count()

    process_file_parallel(
        input_path=args.input,
        output_path=args.output,
        keep_isomers=args.keep_isomers,
        chunksize=args.chunksize,
        num_workers=num_workers,
    )
