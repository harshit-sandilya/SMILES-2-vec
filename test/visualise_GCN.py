import sys
from pathlib import Path

# =====================================================
# Fix Python path (IMPORTANT)
# =====================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import os
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap

from train.utils import has_max_64_atoms

# =====================================================
# Paths
# =====================================================
BASE_DATA_DIR = "data"
BASE_RESULTS_DIR = "results"

EMBEDDINGS_FILE = os.path.join(
    BASE_RESULTS_DIR,
    "embeddings",
    "gcn_embeddings_10k.npy",
)

PLOTS_DIR = os.path.join(BASE_RESULTS_DIR, "plots_GCN")
os.makedirs(PLOTS_DIR, exist_ok=True)

# =====================================================
# Args
# =====================================================
parser = argparse.ArgumentParser()
parser.add_argument(
    "--data",
    type=str,
    default="canonical_smiles_subset_10k.csv",
    help="CSV file with canonical SMILES (relative to data/)",
)
args = parser.parse_args()

# =====================================================
# Resolve data file
# =====================================================
data_file = (
    args.data
    if os.path.isabs(args.data)
    else os.path.join(BASE_DATA_DIR, args.data)
)

if not os.path.exists(data_file):
    raise FileNotFoundError(f"❌ Data file not found: {data_file}")

print(f"📄 Data file       : {data_file}")
print(f"🧠 Embeddings file : {EMBEDDINGS_FILE}")

# =====================================================
# Load embeddings
# =====================================================
if not os.path.exists(EMBEDDINGS_FILE):
    raise FileNotFoundError(
        f"❌ GCN embeddings not found at: {EMBEDDINGS_FILE}"
    )

embeddings = np.load(EMBEDDINGS_FILE)
print(f"✅ Loaded embeddings: {embeddings.shape}")

# =====================================================
# Load dataset + filter
# =====================================================
df = pd.read_csv(data_file)

if "smiles" not in df.columns:
    raise KeyError("❌ CSV must contain a 'smiles' column")

df["smiles"] = df["smiles"].astype(str)

print("\nFiltering molecules with <= 64 atoms...")
df["valid_smiles"] = df["smiles"].apply(has_max_64_atoms)
df = df[df["valid_smiles"]].drop(columns=["valid_smiles"]).reset_index(drop=True)

print(f"🧪 Valid molecules after filter: {len(df)}")

# =====================================================
# Sanity check alignment
# =====================================================
if len(df) != embeddings.shape[0]:
    raise RuntimeError(
        f"❌ Mismatch between filtered molecules ({len(df)}) "
        f"and embeddings ({embeddings.shape[0]})."
    )

print("✅ Embeddings and dataset are aligned")

# =====================================================
# UMAP
# =====================================================
print("\n🔄 Running UMAP...")
reducer = umap.UMAP(
    n_neighbors=15,
    min_dist=0.1,
    n_components=2,
    random_state=42,
)

embedding_2d = reducer.fit_transform(embeddings)
print("✅ UMAP completed:", embedding_2d.shape)

# =====================================================
# Plotting
# =====================================================
plt.style.use("seaborn-v0_8-whitegrid")

properties = [
    ("molecular_weight", "Molecular Weight"),
    ("logp", "LogP"),
    ("hbd", "HBD"),
    ("hba", "HBA"),
]

print("\n🎨 Starting plotting loop...")

for prop, label in properties:
    if prop not in df.columns:
        print(f"⚠️ Skipping {prop} (not found in dataset)")
        continue

    print(f"🖼️ Plotting: {prop}")

    fig, ax = plt.subplots(figsize=(12, 10))

    scatter = ax.scatter(
        embedding_2d[:, 0],
        embedding_2d[:, 1],
        c=df[prop],
        cmap="viridis",
        s=10,
        alpha=0.7,
    )

    cbar = plt.colorbar(scatter)
    cbar.set_label(label, rotation=270, labelpad=15)

    ax.set_title(
        f"GCN Embeddings (10k), colored by {label}",
        fontsize=16,
    )
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")

    out_file = os.path.join(PLOTS_DIR, f"umap_gcn_10k_{prop}.png")
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"✅ Saved: {out_file}")

# =====================================================
# Final confirmation
# =====================================================
print("\n📂 Files in results/plots_GCN:")
print(os.listdir(PLOTS_DIR))
print("🎉 GCN visualization pipeline completed successfully.")
