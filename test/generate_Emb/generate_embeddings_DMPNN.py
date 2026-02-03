import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np
from tqdm import tqdm
from torch_geometric.loader import DataLoader

from train.lightning_model_DMPNN import GraphMoleculeLightningDMPNN
from preprocess.dataset import MaskedMoleculeDataset
from preprocess.tokenizer import SMILESTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = GraphMoleculeLightningDMPNN.load_from_checkpoint(
    "results/models/final_model_DMPNN.ckpt"
).model.to(device)
model.eval()

tokenizer = SMILESTokenizer()
smiles = open("data/canonical_smiles_subset_10k.csv").read().splitlines()[1:]
tokenized = [tokenizer.tokenize(s) for s in smiles]

dataset = MaskedMoleculeDataset(tokenized, 0.0, 0.0)
loader = DataLoader(dataset, batch_size=64, shuffle=False)

all_emb = []

with torch.no_grad():
    for batch in tqdm(loader):
        batch = batch.to(device)
        emb = model(batch)
        all_emb.append(emb.cpu().numpy())

embeddings = np.concatenate(all_emb, axis=0)
np.save("results/embeddings/dmpnn_embeddings_10k.npy", embeddings)

print("Saved DMPNN embeddings:", embeddings.shape)
