import argparse

import pandas as pd
from rdkit import Chem


def canonicalize_smiles(smiles: str, keep_isomers: bool = False) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=keep_isomers)
    return None


def process_csv(file_path: str, keep_isomers: bool = False):
    df = pd.read_csv(file_path)
    if "smiles" not in df.columns:
        raise ValueError("CSV must have a 'smiles' column.")
    df["smiles"] = df["smiles"].apply(lambda s: canonicalize_smiles(s, keep_isomers))
    df = df.dropna(subset=["smiles"])
    df.to_csv(file_path, index=False)
    print(f"[✔] Canonical SMILES saved to {file_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Canonicalize SMILES in a CSV.")
    parser.add_argument("file", type=str, help="Path to the input CSV file")
    parser.add_argument(
        "--keep-isomers",
        action="store_true",
        help="Preserve stereochemistry and isotopes",
    )
    args = parser.parse_args()
    process_csv(args.file, keep_isomers=args.keep_isomers)
