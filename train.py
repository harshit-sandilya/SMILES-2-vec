import os
from argparse import ArgumentParser

import pandas as pd
import torch
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from config import *
from dataset import MaskedMoleculeDataset
from model import GraphMoleculeModel
from tokenizer import SMILESTokenizer
from utils import has_max_64_atoms

if not os.path.exists("model"):
    os.makedirs("model")

parser = ArgumentParser()
parser.add_argument(
    "--train-file", type=str, required=True, help="Path to the training CSV file"
)
args = parser.parse_args()
train_file = args.train_file

df = pd.read_csv(train_file)
df["valid_smiles"] = df["smiles"].apply(has_max_64_atoms)
df_filtered = (
    df[df["valid_smiles"]].drop(columns=["valid_smiles"]).reset_index(drop=True)
)
smiles = df_filtered["smiles"].values
tokenizer = SMILESTokenizer()
X = [tokenizer.tokenize(s) for s in smiles]
dataset = MaskedMoleculeDataset(X)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

model = GraphMoleculeModel(
    hidden_dim=hidden_dim,
    num_layers=num_layers,
    num_heads=num_heads,
    ATOM_VOCAB_SIZE=ATOM_VOCAB_SIZE,
    BOND_VOCAB_SIZE=BOND_VOCAB_SIZE,
    device=device,
)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
loss_fn_atom = torch.nn.CrossEntropyLoss(ignore_index=-1)
loss_fn_bond = torch.nn.CrossEntropyLoss(ignore_index=-1)

for epoch in range(EPOCHS):
    model.train()
    total_atom_loss = 0
    total_bond_loss = 0

    progress_bar = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

    for batch in progress_bar:
        batch = batch.to(device)
        optimizer.zero_grad()

        atom_logits, bond_logits = model(batch)

        true_atoms = batch.y_atoms
        true_bonds = batch.y_bonds

        atom_loss = loss_fn_atom(atom_logits, true_atoms)

        masked_bond_mask = true_bonds != -1
        bond_loss = loss_fn_bond(
            bond_logits[masked_bond_mask], true_bonds[masked_bond_mask]
        )

        if torch.isnan(bond_loss):
            bond_loss = 0.0

        total_loss = atom_loss + bond_loss

        total_loss.backward()
        optimizer.step()

        total_atom_loss += atom_loss.item()
        if isinstance(bond_loss, torch.Tensor):
            total_bond_loss += bond_loss.item()

        progress_bar.set_postfix(
            {
                "atom_loss": f"{atom_loss.item():.4f}",
                "bond_loss": f"{bond_loss.item() if isinstance(bond_loss, torch.Tensor) else 0.0:.4f}",
            }
        )

    avg_atom_loss = total_atom_loss / len(loader)
    avg_bond_loss = total_bond_loss / len(loader)
    print(
        f"Epoch {epoch+1} Summary: Avg Atom Loss: {avg_atom_loss:.4f}, Avg Bond Loss: {avg_bond_loss:.4f}"
    )

    if (epoch + 1) % 10 == 0:
        torch.save(model.state_dict(), f"model/model_epoch_{epoch+1}.pt")

print("Training finished.")
torch.save(model.state_dict(), "model/model_final.pt")
