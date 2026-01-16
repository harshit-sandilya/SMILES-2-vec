import os
import json
import numpy as np
import torch
import umap
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

from lightning.data import StreamingDataset
from train.lightning_model_GIN import GraphMoleculeLightningGIN


BASE_DATA_DIR = "data"
BASE_RESULTS_DIR = "results"


def plot_colored(embeddings_2d, values, name, output_dir):
    values = values.to_numpy()

    # Check for invalid UMAP coordinates
    valid_coords = ~(np.isnan(embeddings_2d[:, 0]) | np.isnan(embeddings_2d[:, 1]) | 
                     np.isinf(embeddings_2d[:, 0]) | np.isinf(embeddings_2d[:, 1]))
    
    print(f"\n{name}:")
    print(f"  Valid UMAP coordinates: {valid_coords.sum()}/{len(valid_coords)}")
    
    # Filter to only valid coordinates
    embeddings_2d_valid = embeddings_2d[valid_coords]
    values_valid = values[valid_coords]
    
    # mask for valid (non-NaN) property values among valid coordinates
    valid_mask = ~np.isnan(values_valid)
    
    print(f"  Property values: {valid_mask.sum()} valid, {(~valid_mask).sum()} missing")

    plt.figure(figsize=(10, 8))

    # 1) plot molecules WITHOUT the property (greyed out) - MORE VISIBLE
    if (~valid_mask).sum() > 0:
        plt.scatter(
            embeddings_2d_valid[~valid_mask, 0],
            embeddings_2d_valid[~valid_mask, 1],
            s=15,
            alpha=0.6,
            color="lightgrey",
            label=f"missing value (n={(~valid_mask).sum()})",
            edgecolors='none'
        )

    # 2) plot molecules WITH the property (colored)
    if valid_mask.sum() > 0:
        sc = plt.scatter(
            embeddings_2d_valid[valid_mask, 0],
            embeddings_2d_valid[valid_mask, 1],
            c=values_valid[valid_mask],
            cmap="viridis",
            s=15,
            alpha=0.8,
            label=f"available value (n={valid_mask.sum()})",
            edgecolors='none'
        )
        plt.colorbar(sc, label=name)

    plt.title(f"GIN Embeddings colored by {name}\n({len(embeddings_2d_valid)} molecules plotted)")
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.legend(markerscale=1.5, frameon=True, loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"umap_{name}.png"), dpi=300)
    plt.close()



def main():
    model_name = "GIN"
    output_dir = os.path.join(BASE_RESULTS_DIR, f"plots_{model_name}")
    os.makedirs(output_dir, exist_ok=True)

    # ---------------- Load model ----------------
    checkpoint_path = os.path.join(
        BASE_RESULTS_DIR, "models", "final_model_GIN.ckpt"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    lightning_model = GraphMoleculeLightningGIN.load_from_checkpoint(
        checkpoint_path
    )
    model = lightning_model.model
    model.to(device)
    model.eval()

    print("Loaded GIN model for visualization.")

    # ---------------- Load FULL optimized dataset ----------------
    full_dataset = StreamingDataset(
        input_dir=os.path.join(BASE_DATA_DIR, "optimized_graph_dataset")
    )

    all_embeddings = []

    with torch.no_grad():
        for data in tqdm(full_dataset, desc="Extracting embeddings"):
            data = data.to(device)

            # create a fake batch index (single-graph batch)
            data.batch = torch.zeros(
                data.x.size(0), dtype=torch.long, device=data.x.device
            )

            emb = model.get_embedding(data)
            all_embeddings.append(emb.cpu().numpy())

    embeddings = np.concatenate(all_embeddings, axis=0)

    print(f"✔ Extracted {embeddings.shape[0]} embeddings of dimension {embeddings.shape[1]}")
    
    # Check embedding statistics
    print(f"\nEmbedding statistics:")
    print(f"  Mean: {embeddings.mean():.4f}")
    print(f"  Std: {embeddings.std():.4f}")
    print(f"  Min: {embeddings.min():.4f}")
    print(f"  Max: {embeddings.max():.4f}")
    print(f"  Contains NaN: {np.isnan(embeddings).any()}")
    print(f"  Contains Inf: {np.isinf(embeddings).any()}")

    np.save(os.path.join(output_dir, "embeddings.npy"), embeddings)

    # ---------------- Load molecular properties ----------------
    props_path = os.path.join(BASE_DATA_DIR, "canonical_smiles_subset_10k.csv")

    if not os.path.exists(props_path):
        raise FileNotFoundError(
            f"Property CSV not found at {props_path}"
        )

    props_df = pd.read_csv(props_path)

    # NOTE: assumes same order after preprocessing (acceptable for visualization)
    props_df = props_df.iloc[: len(embeddings)]

    print(f"\nNumber of embeddings: {len(embeddings)}")
    print(f"Number of property rows: {len(props_df)}")
    
    # Debug: check for NaN values in each property
    print("\nProperty completeness:")
    for prop in ["molecular_weight", "logp", "hba", "hbd"]:
        if prop in props_df.columns:
            valid_count = props_df[prop].notna().sum()
            print(f"  {prop}: {valid_count}/{len(props_df)} ({100*valid_count/len(props_df):.1f}%)")
        else:
            print(f"  {prop}: COLUMN NOT FOUND")

    # ---------------- UMAP with better parameters ----------------
    print("\nRunning UMAP dimensionality reduction...")
    print("This may take a few minutes...")
    
    # Use more aggressive UMAP parameters to handle difficult embeddings
    reducer = umap.UMAP(
        n_neighbors=30,  # increased from 15
        min_dist=0.0,    # reduced from 0.1 for tighter clusters
        n_components=2,
        metric='euclidean',
        random_state=42,
        n_epochs=500,    # more training iterations
        init='spectral',
        verbose=True
    )
    
    try:
        embeddings_2d = reducer.fit_transform(embeddings)
    except Exception as e:
        print(f"\n⚠ UMAP failed with error: {e}")
        print("Falling back to PCA for visualization...")
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2, random_state=42)
        embeddings_2d = pca.fit_transform(embeddings)
        print(f"PCA explained variance: {pca.explained_variance_ratio_.sum():.2%}")
    
    # Check for invalid coordinates
    valid_coords = ~(np.isnan(embeddings_2d[:, 0]) | np.isnan(embeddings_2d[:, 1]) | 
                     np.isinf(embeddings_2d[:, 0]) | np.isinf(embeddings_2d[:, 1]))
    
    print(f"\nUMAP Results:")
    print(f"  Valid coordinates: {valid_coords.sum()}/{len(valid_coords)}")
    print(f"  Invalid coordinates: {(~valid_coords).sum()}")
    
    if valid_coords.sum() < 10:
        print("\n⚠ WARNING: Very few valid UMAP coordinates!")
        print("This suggests the embeddings may not have enough variation.")
        print("Consider checking your model's embedding layer.")

    # ---------------- Plain plot (only valid coordinates) ----------------
    embeddings_2d_valid = embeddings_2d[valid_coords]
    
    plt.figure(figsize=(10, 8))
    plt.scatter(
        embeddings_2d_valid[:, 0],
        embeddings_2d_valid[:, 1],
        s=15,
        alpha=0.6,
        edgecolors='none'
    )
    plt.title(f"GIN Molecular Embeddings (UMAP)\n{len(embeddings_2d_valid)} molecules")
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "umap.png"), dpi=300)
    plt.close()
    
    print(f"\n✔ Saved plain UMAP plot with {len(embeddings_2d_valid)} points")

    # ---------------- Property-colored plots ----------------
    plot_colored(
        embeddings_2d,
        props_df["molecular_weight"],
        "molecular_weight",
        output_dir,
    )

    plot_colored(
        embeddings_2d,
        props_df["logp"],
        "logp",
        output_dir,
    )

    plot_colored(
        embeddings_2d,
        props_df["hba"],
        "hba",
        output_dir,
    )

    plot_colored(
        embeddings_2d,
        props_df["hbd"],
        "hbd",
        output_dir,
    )

    # ---------------- Metadata ----------------
    metadata = {
        "model": model_name,
        "num_molecules": int(embeddings.shape[0]),
        "num_valid_umap_coords": int(valid_coords.sum()),
        "embedding_dim": int(embeddings.shape[1]),
        "checkpoint": checkpoint_path,
        "properties_used": ["molecular_weight", "logp", "hba", "hbd"],
    }

    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n✔ All plots and embeddings saved to {output_dir}")


if __name__ == "__main__":
    main()
