import os
import glob
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
EMBEDDINGS_DIR = os.path.join(BASE_RESULTS_DIR, "embeddings")
PLOTS_DIR = os.path.join(BASE_RESULTS_DIR, "plots")

os.makedirs(PLOTS_DIR, exist_ok=True)

# =====================================================
# Args (kept optional for reproducibility)
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

print(f"📄 Data file      : {data_file}")

# =====================================================
# Load embeddings (merged OR chunks)
# =====================================================
merged_embeddings_path = os.path.join(EMBEDDINGS_DIR, "molecule_embeddings.pkl")

if os.path.exists(merged_embeddings_path):
    print("📦 Loading merged embeddings file...")
    results_df = pd.read_pickle(merged_embeddings_path)
else:
    chunk_files = sorted(glob.glob(os.path.join(EMBEDDINGS_DIR, "chunk_*.pkl")))
    if not chunk_files:
        raise FileNotFoundError(
            "❌ No embeddings found in results/embeddings/. "
            "Expected molecule_embeddings.pkl or chunk_*.pkl files."
        )

    print(f"🔗 Merging {len(chunk_files)} embedding chunks...")
    results_df = pd.concat(
        [pd.read_pickle(f) for f in chunk_files],
        ignore_index=True,
    )

print(f"✅ Loaded embeddings: {len(results_df)}")

# =====================================================
# Load dataset + filter
# =====================================================
df = pd.read_csv(data_file)

if "smiles" not in df.columns:
    raise KeyError("❌ CSV must contain a 'smiles' column")

df["smiles"] = df["smiles"].astype(str)
results_df["smiles"] = results_df["smiles"].astype(str)

df["valid_smiles"] = df["smiles"].apply(has_max_64_atoms)
df = df[df["valid_smiles"]].drop(columns=["valid_smiles"]).reset_index(drop=True)

print(f"🧪 Valid molecules after filter: {len(df)}")

# =====================================================
# Merge embeddings with properties
# =====================================================
merged_df = pd.merge(df, results_df, on="smiles", how="inner")

print(f"🔗 Merged dataset size: {len(merged_df)}")

if len(merged_df) == 0:
    raise RuntimeError(
        "❌ merged_df is EMPTY.\n"
        "This means SMILES in embeddings and CSV do not match.\n"
        "Check canonicalization consistency."
    )

# =====================================================
# Prepare embedding matrix
# =====================================================
if "embedding" not in merged_df.columns:
    raise KeyError("❌ Embeddings file must contain an 'embedding' column")

embeddings_matrix = np.stack(merged_df["embedding"].values)
print("🧠 Embedding matrix shape:", embeddings_matrix.shape)

# =====================================================
# UMAP
# =====================================================
print("🔄 Running UMAP...")
reducer = umap.UMAP(
    n_neighbors=15,
    min_dist=0.1,
    n_components=2,
    random_state=42,
)

embedding_2d = reducer.fit_transform(embeddings_matrix)
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

print("🎨 Starting plotting loop...")

for prop, label in properties:
    if prop not in merged_df.columns:
        print(f"⚠️ Skipping {prop} (not found in dataset)")
        continue

    print(f"🖼️ Plotting: {prop}")

    fig, ax = plt.subplots(figsize=(12, 10))

    scatter = ax.scatter(
        embedding_2d[:, 0],
        embedding_2d[:, 1],
        c=merged_df[prop],
        cmap="viridis",
        s=10,
        alpha=0.7,
    )

    cbar = plt.colorbar(scatter)
    cbar.set_label(label, rotation=270, labelpad=15)

    ax.set_title(
        f"UMAP of SMILES-2-Vec Embeddings (10k), colored by {label}",
        fontsize=16,
    )
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")

    out_file = os.path.join(PLOTS_DIR, f"umap_10k_{prop}.png")
    plt.savefig(out_file, dpi=300)
    plt.close()

    print(f"✅ Saved: {out_file}")

# =====================================================
# Final confirmation
# =====================================================
print("📂 Files in results/plots:")
print(os.listdir(PLOTS_DIR))
print("🎉 Visualization pipeline completed successfully.")
