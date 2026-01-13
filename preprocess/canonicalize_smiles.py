import argparse
from functools import partial
from multiprocessing import Pool, cpu_count
from pathlib import Path

import pandas as pd
from rdkit import Chem
from tqdm import tqdm


# ==============================
# Resolve project root and data dir
# ==============================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"


# ==============================
# Canonicalization function
# ==============================

def canonicalize_smiles(smiles: str, isomeric: bool) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        return Chem.MolToSmiles(
            mol,
            canonical=True,
            isomericSmiles=isomeric
        )
    return None


# ==============================
# Parallel processing logic
# ==============================

def process_file_parallel(
    input_filename: str,
    output_filename: str,
    keep_isomers: bool,
    chunksize: int,
    num_workers: int,
):
    input_path = DATA_DIR / input_filename
    output_path = DATA_DIR / output_filename

    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        total_rows = sum(1 for _ in open(input_path, encoding="utf-8")) - 1
        if total_rows <= 0:
            print(f"Error: {input_path} is empty or has only a header.")
            return
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_path}")
        return

    print(f"Input file  : {input_path}")
    print(f"Output file : {output_path}")
    print(f"Total rows  : {total_rows}")
    print(f"Workers     : {num_workers}")

    reader = pd.read_csv(input_path, chunksize=chunksize)
    worker_func = partial(canonicalize_smiles, isomeric=keep_isomers)

    header_written = False
    processed_rows = 0

    with Pool(processes=num_workers) as pool:
        with tqdm(total=total_rows, desc="Canonicalizing SMILES", unit="mol") as pbar:
            for chunk in reader:
                if "smiles" not in chunk.columns:
                    raise ValueError("Input CSV must contain a 'smiles' column.")

                canonical_results = pool.map(worker_func, chunk["smiles"])
                chunk["smiles"] = canonical_results
                chunk.dropna(subset=["smiles"], inplace=True)

                if not chunk.empty:
                    mode = "w" if not header_written else "a"
                    chunk.to_csv(
                        output_path,
                        index=False,
                        mode=mode,
                        header=not header_written,
                    )
                    header_written = True

                pbar.update(len(chunk))
                processed_rows += len(chunk)

    print("-" * 40)
    print("✔ Canonicalization complete")
    print(f"Valid molecules processed: {processed_rows}")
    print(f"Saved to: {output_path}")
    print("-" * 40)


# ==============================
# CLI
# ==============================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Canonicalize SMILES using parallel processing",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "--input",
        type=str,
        default="canonical_smiles_subset_10k.csv",
        help="Input CSV filename inside data/ (must contain 'smiles')",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="canonical_smiles_subset_10k_canonical.csv",
        help="Output CSV filename (saved inside data/)",
    )

    parser.add_argument(
        "--keep-isomers",
        action="store_true",
        help="Preserve stereochemistry and isotope information",
    )

    parser.add_argument(
        "--chunksize",
        type=int,
        default=5000,
        help="Number of SMILES per chunk (default: 5000)",
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of CPU cores to use (default: all available)",
    )

    args = parser.parse_args()
    num_workers = args.num_workers or cpu_count()

    process_file_parallel(
        input_filename=args.input,
        output_filename=args.output,
        keep_isomers=args.keep_isomers,
        chunksize=args.chunksize,
        num_workers=num_workers,
    )
