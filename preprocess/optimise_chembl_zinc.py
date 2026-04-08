import sys
from pathlib import Path
import warnings
import pandas as pd
import torch
from litdata import optimize

from preprocess.tokenize import SMILESTokenizer
from train.utils import create_masked_graph_from_tensors

warnings.filterwarnings(
    "ignore",
    message="An item was larger than the target chunk size",
    category=UserWarning,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"


def process_single_smiles(smiles: str, tokenizer):
    """Process one SMILES string into a masked graph (safe version)."""
    try:
        tokenized_smiles = tokenizer.tokenize(smiles)

        atomic_numbers = torch.tensor(
            tokenized_smiles["atomic_numbers"], dtype=torch.long
        )
        bond_matrix = torch.tensor(
            tokenized_smiles["bond_matrix"], dtype=torch.float
        )

        data = create_masked_graph_from_tensors(
            atomic_numbers=atomic_numbers,
            bond_matrix=bond_matrix,
            mask_ratio_atoms=0.15,
            mask_ratio_bonds=0.15,
        )

        return {
            "x": data.x,
            "edge_index": data.edge_index,
            "edge_attr": data.edge_attr,
            "y_atoms": data.y_atoms,
            "y_bonds": data.y_bonds,
        }

    except Exception as e:
        return None

from torch.utils.data import Dataset

class MaskedMoleculeDataset(Dataset):
    def __init__(self, tokenized_smiles_list, mask_ratio_atoms=0.0, mask_ratio_bonds=0.0):
        self.data = tokenized_smiles_list
        self.mask_ratio_atoms = mask_ratio_atoms
        self.mask_ratio_bonds = mask_ratio_bonds

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tokenized = self.data[idx]

        atomic_numbers = torch.tensor(tokenized["atomic_numbers"], dtype=torch.long)
        bond_matrix = torch.tensor(tokenized["bond_matrix"], dtype=torch.float)

        return create_masked_graph_from_tensors(
            atomic_numbers,
            bond_matrix,
            self.mask_ratio_atoms,
            self.mask_ratio_bonds,
        )
        
def parallel_process_and_create_graphs(input_file: str):
    """
    Generator function called by litdata workers.
    Handles both ZINC and ChEMBL CSV formats:
      - ZINC:   columns [zinc_id,   smiles]
      - ChEMBL: columns [chembl_id, smiles]
    Both share the 'smiles' column name so no special branching needed.
    """
    tokenizer = SMILESTokenizer()
    csv_chunk_size = 10_000

    source_name = Path(input_file).stem  # e.g. "zinc_smiles" or "chembl_smiles"
    print(f"[Worker] Processing source: {source_name}")

    for chunk_df in pd.read_csv(input_file, chunksize=csv_chunk_size):
        if "smiles" not in chunk_df.columns:
            raise ValueError(
                f"No 'smiles' column found in {input_file}. "
                f"Available columns: {chunk_df.columns.tolist()}"
            )

        smiles_list = chunk_df["smiles"].dropna().tolist()
        for smiles in smiles_list:
            data = process_single_smiles(smiles, tokenizer)
            if data is not None:
                yield data


if __name__ == "__main__":
    # ── Input files ───────────────────────────────────────────────────────────
    zinc_csv    = DATA_DIR / "zinc_smiles.csv"
    chembl_csv  = DATA_DIR / "chembl_smiles.csv"
    output_dir  = DATA_DIR / "optimized_graph_dataset"

    # Validate inputs exist
    input_files = []
    for path in [zinc_csv, chembl_csv]:
        if path.exists():
            input_files.append(str(path))
            print(f"✔ Found: {path}")
        else:
            print(f"✘ Missing (skipping): {path}")

    if not input_files:
        raise FileNotFoundError(
            "No input CSV files found. "
            "Run the downloader script first to generate zinc_smiles.csv and chembl_smiles.csv."
        )

    print(f"\nStarting dataset optimization with {len(input_files)} source(s)...")
    print(f"Output directory: {output_dir}\n")

    # ── litdata optimize ──────────────────────────────────────────────────────
    # litdata will call parallel_process_and_create_graphs once per input file,
    # distributing files across workers automatically.
    optimize(
        fn=parallel_process_and_create_graphs,
        inputs=input_files,          # [zinc_smiles.csv, chembl_smiles.csv]
        output_dir=str(output_dir),
        num_workers=4,
        chunk_bytes="64MB",
        mode="overwrite",
    )

    print("\n✔ Dataset optimization complete!")
    print(f"Streamable dataset saved at: {output_dir}")