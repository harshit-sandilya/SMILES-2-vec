import argparse
import csv
from multiprocessing import Pool, cpu_count

from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski
from tqdm import tqdm

DEFAULT_INPUT_CSV = "data/zinc_smiles.csv"
DEFAULT_OUTPUT_CSV = "data/smiles.csv"
DEFAULT_CHUNK_SIZE = 10000


def compute_properties_row(row):
    if len(row) != 2:
        return None
    zinc_id, smiles = row
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        return [
            zinc_id,
            smiles,
            Descriptors.MolWt(mol),
            Crippen.MolLogP(mol),
            Lipinski.NumHDonors(mol),
            Lipinski.NumHAcceptors(mol),
        ]
    except Exception:
        return None


def chunked_iterable(iterable, size):
    chunk = []
    for item in iterable:
        chunk.append(item)
        if len(chunk) == size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def count_data_rows(file_path):
    with open(file_path, "r") as f:
        return sum(1 for line in f) - 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate ZINC metrics for SMILES.")
    parser.add_argument(
        "--input_csv", type=str, default=DEFAULT_INPUT_CSV, help="Input CSV file path"
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default=DEFAULT_OUTPUT_CSV,
        help="Output CSV file path",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help="Chunk size for processing",
    )
    args = parser.parse_args()

    total_rows = count_data_rows(args.input_csv)
    if total_rows <= 0:
        print("CSV is empty or contains only a header.")
    else:
        with open(args.input_csv, "r") as infile, open(
            args.output_csv, "w", newline=""
        ) as outfile:
            reader = csv.reader(infile)
            writer = csv.writer(outfile)

            try:
                header = next(reader)
            except StopIteration:
                print("CSV is empty.")
                exit()

            writer.writerow(
                ["zinc_id", "smiles", "molecular_weight", "logp", "hbd", "hba"]
            )

            with Pool(processes=cpu_count()) as pool:
                for chunk in tqdm(
                    chunked_iterable(reader, args.chunk_size),
                    total=(total_rows // args.chunk_size) + 1,
                    desc="Processing molecules",
                    unit="chunk",
                ):
                    results = pool.map(compute_properties_row, chunk)
                    for row in results:
                        if row:
                            writer.writerow(row)
