import os
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import umap

from utils import has_max_64_atoms

if not os.path.exists("results"):
    os.makedirs("results")

parser = ArgumentParser()
parser.add_argument(
    "--embeddings",
    type=str,
    required=True,
    help="Path to the generated embeddings file",
)
parser.add_argument(
    "--data", type=str, required=True, help="Path to the original data file for labels"
)
args = parser.parse_args()
embedding_file = args.embeddings
data_file = args.data

results_df = pd.read_pickle(embedding_file)
print(f"Loaded {len(results_df)} embeddings from pickle file.")
df = pd.read_csv(data_file)
df["valid_smiles"] = df["smiles"].apply(has_max_64_atoms)
df_filtered = (
    df[df["valid_smiles"]].drop(columns=["valid_smiles"]).reset_index(drop=True)
)
merged_df = pd.merge(df_filtered, results_df, on="smiles", how="inner")
embeddings_matrix = np.stack(merged_df["embedding"].values)

reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
print("Running UMAP... (This may take a minute)")
embedding_2d = reducer.fit_transform(embeddings_matrix)

plt.style.use("seaborn-v0_8-whitegrid")
properties = [
    ("molecular_weight", "Molecular Weight"),
    ("logp", "LogP"),
    ("hbd", "HBD"),
    ("hba", "HBA"),
]
for prop, label in properties:
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
        f"2D UMAP Projection of Molecule Embeddings, Colored by {label}", fontsize=16
    )
    ax.set_xlabel("UMAP Dimension 1")
    ax.set_ylabel("UMAP Dimension 2")
    plt.savefig(f"results/umap_visualization_{prop}.png")
    plt.close()
