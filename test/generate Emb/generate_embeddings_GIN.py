import sys
from pathlib import Path

import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
from torch_geometric.loader import DataLoader

# ============================================================
# Fix Python path
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from preprocess.tokenizer import SMILESTokenizer
from preprocess.dataset import MaskedMoleculeDataset
from train.lightning_model_GIN import GraphMoleculeLightningGIN
from train.utils import has_max_64_atoms

# ============================================================
# Paths
# ============================================================
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
EMB_DIR = RESULTS_DIR / "embeddings"
EMB_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# Config
# ============================================================
BATCH_SIZE = 256
DATA_FILE = DATA_DIR / "canonical_smiles_subset_10k.csv"
OUTPUT_FILE = EMB_DIR / "gin_embeddings_10k.npy"


def main():
    print("=" * 70)
    print("GIN MOLECULAR EMBEDDING GENERATION")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # ============================================================
    # Load & filter data
    # ============================================================
    df = pd.read_csv(DATA_FILE)
    df["valid_smiles"] = df["smiles"].apply(has_max_64_atoms)
    df = df[df["valid_smiles"]].copy()

    smiles_list = df["smiles"].tolist()

    # ============================================================
    # Tokenization
    # ============================================================
    tokenizer = SMILESTokenizer()
    tokenized = [
        tokenizer.tokenize(s)
        for s in tqdm(smiles_list, desc="Tokenizing")
    ]

    # ============================================================
    # Dataset (NO masking)
    # ============================================================
    dataset = MaskedMoleculeDataset(
        tokenized_list=tokenized,
        mask_ratio_atoms=0.0,
        mask_ratio_bonds=0.0,
        apply_masking=False,
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
    )

    # ============================================================
    # Model (NO checkpoint loading)
    # ============================================================
    print("\nInitializing GIN model...")
    lightning_model = GraphMoleculeLightningGIN(
        hidden_dim=128,
        num_layers=3,
    ).to(device)

    lightning_model.eval()

    # ============================================================
    # Generate embeddings
    # ============================================================
    print("\nGenerating embeddings...")
    all_embeddings = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Embedding batches"):
            batch = batch.to(device)
            emb = lightning_model.model.get_graph_embedding(batch)
            all_embeddings.append(emb.cpu().numpy())

    embeddings = np.concatenate(all_embeddings, axis=0)

    print(f"\nFinal embeddings shape: {embeddings.shape}")

    # ============================================================
    # Save embeddings
    # ============================================================
    np.save(OUTPUT_FILE, embeddings)
    print(f"✅ Saved embeddings to: {OUTPUT_FILE}")

    print("\nGIN embedding generation completed successfully.")


if __name__ == "__main__":
    main()
