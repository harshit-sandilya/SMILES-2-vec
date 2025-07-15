import os
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from config import *
from dataset import MaskedMoleculeDataset
from model import GraphMoleculeModel
from tokenizer import SMILESTokenizer
from utils import has_max_64_atoms

if not os.path.exists("embeddings"):
    os.makedirs("embeddings")

parser = ArgumentParser()
parser.add_argument(
    "--model-file", type=str, required=True, help="Path to the final model file"
)
parser.add_argument(
    "--data-file", type=str, required=True, help="Path to the data file"
)
args = parser.parse_args()
model_file = args.model_file
data_file = args.data_file

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
inference_model = GraphMoleculeModel(
    hidden_dim=hidden_dim,
    num_layers=num_layers,
    num_heads=num_heads,
    ATOM_VOCAB_SIZE=ATOM_VOCAB_SIZE,
    BOND_VOCAB_SIZE=BOND_VOCAB_SIZE,
    device=device,
)
inference_model.load_state_dict(torch.load(model_file, map_location=device))
inference_model.eval()
print("Model weights loaded successfully.")

df = pd.read_csv(data_file)
df["valid_smiles"] = df["smiles"].apply(has_max_64_atoms)
df_filtered = (
    df[df["valid_smiles"]].drop(columns=["valid_smiles"]).reset_index(drop=True)
)
smiles = df_filtered["smiles"].values
tokenizer = SMILESTokenizer()
X = [tokenizer.tokenize(s) for s in smiles]
dataset = MaskedMoleculeDataset(X, mask_ratio_atoms=0.0, mask_ratio_bonds=0.0)
loader = DataLoader(dataset, batch_size=64, shuffle=True)

all_embeddings = []
all_smiles = []

with torch.no_grad():
    progress_bar_inference = tqdm(loader, desc="Generating Embeddings")
    for batch in progress_bar_inference:
        batch = batch.to(device)
        embeddings = inference_model.get_embedding(batch)
        all_embeddings.append(embeddings.cpu().numpy())
        all_smiles.extend(batch.smile)

final_embeddings = np.concatenate(all_embeddings, axis=0)

results_df = pd.DataFrame({"smiles": all_smiles, "embedding": list(final_embeddings)})

print(f"\nGenerated {len(results_df)} embeddings.")
print(f"Shape of the final embedding matrix: {final_embeddings.shape}")
print("Example of the first result:")
print(results_df.head())

results_df.to_pickle("embeddings/molecule_embeddings.pkl")
print("\nEmbeddings saved to molecule_embeddings.pkl")
