import os
import sys
from pathlib import Path
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch_geometric.loader import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from train.lightning_model_GIN import GraphMoleculeLightningGIN
from preprocess.dataset import MaskedMoleculeDataset
from preprocess.tokenizer import SMILESTokenizer
from train.utils import has_max_64_atoms


# ================= Paths =================
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
MODEL_PATH = RESULTS_DIR / "models" / "final_model_GIN.ckpt"
OUT_PATH = RESULTS_DIR / "embeddings_GIN.pkl"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # -------- Load model --------
    model = GraphMoleculeLightningGIN.load_from_checkpoint(MODEL_PATH)
    model = model.to(device)
    model.eval()
    print("Loaded model:", MODEL_PATH)

    # -------- Load data --------
    df = pd.read_csv(DATA_DIR / "canonical_smiles_subset_10k.csv")
    df["valid"] = df["smiles"].apply(has_max_64_atoms)
    df = df[df["valid"]].reset_index(drop=True)

    smiles = df["smiles"].tolist()
    tokenizer = SMILESTokenizer()
    tokenized = [tokenizer.tokenize(s) for s in tqdm(smiles)]

    dataset = MaskedMoleculeDataset(
        tokenized,
        mask_ratio_atoms=0,
        mask_ratio_bonds=0,
    )

    loader = DataLoader(dataset, batch_size=64, shuffle=False)

    # -------- Generate embeddings --------
    all_embeddings = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Embedding molecules"):
            batch = batch.to(device)
            emb = model.get_embedding(batch)
            all_embeddings.append(emb.cpu().numpy())

    embeddings = np.concatenate(all_embeddings, axis=0)
    print("Embeddings shape:", embeddings.shape)

    # -------- Save --------
    out_df = pd.DataFrame({
        "smiles": smiles,
        "embedding": list(embeddings)
    })

    out_df.to_pickle(OUT_PATH)
    print("Saved to:", OUT_PATH)


if __name__ == "__main__":
    main()
