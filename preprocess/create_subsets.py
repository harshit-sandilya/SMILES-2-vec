import pandas as pd
from rdkit import Chem
from pathlib import Path

RAW_PATH = "data/raw/chembl.csv"
OUT_DIR = Path("data/subsets")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SIZES = [10_000, 30_000, 50_000, 100_000]
SEED = 42
MAX_ATOMS = 64  # align with your project utils

def is_valid_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return mol is not None and mol.GetNumAtoms() <= MAX_ATOMS

df = pd.read_csv(RAW_PATH)

# basic cleaning
df = df.dropna(subset=["smiles"])
df = df[df["smiles"].apply(is_valid_smiles)]

# shuffle once, reproducibly
df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)

for size in SIZES:
    subset = df.iloc[:size]
    out_path = OUT_DIR / f"chembl_{size}.csv"
    subset.to_csv(out_path, index=False)
    print(f"Saved {out_path}")
