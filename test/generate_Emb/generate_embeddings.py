import sys
from pathlib import Path

# =====================================================
# Fix Python path
# =====================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import os
import numpy as np
import torch
from tqdm import tqdm
import pandas as pd
from torch_geometric.data import Batch

from preprocess.tokenizer import SMILESTokenizer
from preprocess.dataset import MaskedMoleculeDataset
from train.lightning_model import GraphMoleculeLightning
from train.utils import has_max_64_atoms


# =====================================================
# Paths
# =====================================================
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = RESULTS_DIR / "models"
EMB_DIR = RESULTS_DIR / "embeddings"
EMB_DIR.mkdir(exist_ok=True)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # =====================================================
    # Load trained GATv2 model
    # =====================================================
    ckpt_path = MODELS_DIR / "final_model.ckpt"

    lightning_model = GraphMoleculeLightning.load_from_checkpoint(
        str(ckpt_path), strict=False
    )

    model = lightning_model.model.to(device)
    model.eval()

    print("Loaded GATv2 model:", ckpt_path)

    # =====================================================
    # Load SMILES
    # =====================================================
    csv_file = DATA_DIR / "canonical_smiles_subset_10k.csv"
    df = pd.read_csv(csv_file)

    print("Original molecules:", len(df))

    df = df[df["smiles"].apply(has_max_64_atoms)]
    print("After filtering:", len(df))

    smiles_list = df["smiles"].tolist()

    # =====================================================
    # Tokenize → Dataset
    # =====================================================
    tokenizer = SMILESTokenizer()
    tokenized = [tokenizer.tokenize(s) for s in tqdm(smiles_list)]

    dataset = MaskedMoleculeDataset(
        tokenized,
        mask_ratio_atoms=0.0,
        mask_ratio_bonds=0.0,
    )

    print("Dataset size:", len(dataset))

    # =====================================================
    # Generate embeddings
    # =====================================================
    all_embeddings = []

    with torch.no_grad():
        for data in tqdm(dataset, desc="Embedding molecules"):
            data = data.to(device)
            batch = Batch.from_data_list([data])
            emb = model.get_graph_embedding(batch)
            all_embeddings.append(emb.cpu().numpy())

    embeddings = np.concatenate(all_embeddings, axis=0)

    print("Final shape:", embeddings.shape)

    # =====================================================
    # Save
    # =====================================================
    out_file = EMB_DIR / "gat_embeddings_10k.npy"
    np.save(out_file, embeddings)

    print("Saved:", out_file)


if __name__ == "__main__":
    main()
