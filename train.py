import os
from argparse import ArgumentParser

import matplotlib.pyplot as plt
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

if not os.path.exists("results"):
    os.makedirs("results")

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
loader = DataLoader(dataset, batch_size=256, shuffle=True)

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

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total trainable model parameters: {total_params}")

step_atom_losses = []
step_bond_losses = []

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
        if masked_bond_mask.sum() > 0:
            bond_loss = loss_fn_bond(
                bond_logits[masked_bond_mask], true_bonds[masked_bond_mask]
            )
        else:
            bond_loss = torch.tensor(0.0, device=device)

        if torch.isnan(bond_loss):
            bond_loss = torch.tensor(0.0, device=device)

        total_loss = atom_loss + bond_loss

        total_loss.backward()
        optimizer.step()

        atom_loss_item = atom_loss.item()
        bond_loss_item = bond_loss.item()
        step_atom_losses.append(atom_loss_item)
        step_bond_losses.append(bond_loss_item)

        total_atom_loss += atom_loss_item
        total_bond_loss += bond_loss_item

        progress_bar.set_postfix(
            {
                "atom_loss": f"{atom_loss_item:.4f}",
                "bond_loss": f"{bond_loss_item:.4f}",
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


print("Generating and saving separate loss curves...")
window_size = len(loader)
print(f"Using a rolling window size of {window_size} (1 epoch) for smoothing.")

atom_loss_series = pd.Series(step_atom_losses)
bond_loss_series = pd.Series(step_bond_losses)

atom_loss_rolling = atom_loss_series.rolling(window=window_size, min_periods=1).mean()
bond_loss_rolling = bond_loss_series.rolling(window=window_size, min_periods=1).mean()

# --- Plot 1: Atom Loss ---
plt.figure(figsize=(12, 7))
plt.plot(step_atom_losses, label="Step Atom Loss", alpha=0.3, color="tab:blue")
plt.plot(
    atom_loss_rolling,
    label=f"Epoch-wise Rolling Average (window={window_size})",
    linewidth=2,
    color="tab:blue",
)
plt.title("Atom Prediction Loss Curve")
plt.xlabel("Training Step (Batch)")
plt.ylabel("Loss")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.savefig("results/loss_curve_atom.png")
plt.close()
print("Atom loss curve saved to results/loss_curve_atom.png")


# --- Plot 2: Bond Loss ---
plt.figure(figsize=(12, 7))
plt.plot(step_bond_losses, label="Step Bond Loss", alpha=0.3, color="tab:orange")
plt.plot(
    bond_loss_rolling,
    label=f"Epoch-wise Rolling Average (window={window_size})",
    linewidth=2,
    color="tab:orange",
)
plt.title("Bond Prediction Loss Curve")
plt.xlabel("Training Step (Batch)")
plt.ylabel("Loss")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.savefig("results/loss_curve_bond.png")
plt.close()
print("Bond loss curve saved to results/loss_curve_bond.png")
