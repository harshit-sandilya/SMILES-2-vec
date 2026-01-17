import os
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from sklearn.decomposition import PCA
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
from preprocess import dataset
from train import model
from train.lightning_model_GIN import GraphMoleculeLightningGIN
from preprocess.dataset import MaskedMoleculeDataset
from preprocess.tokenizer import SMILESTokenizer
from train.utils import has_max_64_atoms


# ==============================
# Resolve paths
# ==============================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = RESULTS_DIR / "models"


def diagnose_embeddings(embeddings):
    """Diagnose embedding quality."""
    print(f"\n{'='*70}")
    print("EMBEDDING DIAGNOSTICS")
    print(f"{'='*70}")
    
    print(f"Shape: {embeddings.shape}")
    print(f"\nBasic Statistics:")
    print(f"  Mean:   {embeddings.mean():.6f}")
    print(f"  Std:    {embeddings.std():.6f}")
    print(f"  Min:    {embeddings.min():.6f}")
    print(f"  Max:    {embeddings.max():.6f}")
    
    print(f"\nData Quality:")
    print(f"  Contains NaN: {np.isnan(embeddings).any()}")
    print(f"  Contains Inf: {np.isinf(embeddings).any()}")
    print(f"  Fraction zeros: {(embeddings == 0).sum() / embeddings.size:.4%}")
    
    # Check uniqueness
    unique_embeddings = np.unique(embeddings, axis=0)
    print(f"\nUniqueness:")
    print(f"  Unique embeddings: {len(unique_embeddings)}/{len(embeddings)}")
    print(f"  Duplicate rate: {(1 - len(unique_embeddings)/len(embeddings)):.2%}")
    
    # Per-dimension variance
    per_dim_std = embeddings.std(axis=0)
    print(f"\nPer-dimension variance:")
    print(f"  Mean std: {per_dim_std.mean():.6f}")
    print(f"  Dims with std < 0.001: {(per_dim_std < 0.001).sum()}/{len(per_dim_std)}")
    
    # Sample embeddings
    print(f"\nFirst 3 embeddings (first 10 dims):")
    for i in range(min(3, len(embeddings))):
        print(f"  [{i}] {embeddings[i][:10]}")
    
    # Health check
    if embeddings.std() < 0.01:
        print(f"\n❌ CRITICAL: std < 0.01 - embeddings lack variation")
        return False
    if len(unique_embeddings) < len(embeddings) * 0.1:
        print(f"\n❌ CRITICAL: >90% duplicates - model not learning")
        return False
    
    print(f"\n✅ Embeddings look healthy!")
    return True


def plot_colored(embeddings_2d, values, name, output_dir, method="PCA"):
    """Plot embeddings colored by property."""
    values = values.to_numpy()
    valid_mask = ~np.isnan(values)
    
    print(f"  {name}: {valid_mask.sum()} valid, {(~valid_mask).sum()} missing")

    plt.figure(figsize=(12, 9))

    # Plot missing values (grey)
    if (~valid_mask).sum() > 0:
        plt.scatter(
            embeddings_2d[~valid_mask, 0],
            embeddings_2d[~valid_mask, 1],
            s=20,
            alpha=0.3,
            color="lightgrey",
            label=f"missing (n={~valid_mask.sum()})",
            edgecolors='none'
        )

    # Plot available values (colored)
    if valid_mask.sum() > 0:
        sc = plt.scatter(
            embeddings_2d[valid_mask, 0],
            embeddings_2d[valid_mask, 1],
            c=values[valid_mask],
            cmap="viridis",
            s=25,
            alpha=0.7,
            label=f"available (n={valid_mask.sum()})",
            edgecolors='none'
        )
        plt.colorbar(sc, label=name, fraction=0.046, pad=0.04)

    plt.title(f"GIN Embeddings ({method}) - {name}\n{len(embeddings_2d)} molecules")
    plt.xlabel(f"{method}-1")
    plt.ylabel(f"{method}-2")
    plt.legend(markerscale=1.5, frameon=True, loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{method.lower()}_{name}.png"), 
                dpi=300, bbox_inches='tight')
    plt.close()


def main():
    output_dir = RESULTS_DIR / "plots_GIN"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'='*70}")
    print("GIN MOLECULAR EMBEDDING VISUALIZATION")
    print(f"{'='*70}")

    # ============================================================
    # LOAD MODEL (same as generate_embeddings.py)
    # ============================================================
    checkpoint_path = MODELS_DIR / "final_model_GIN.ckpt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    lightning_model = GraphMoleculeLightningGIN.load_from_checkpoint(str(checkpoint_path))
    model = lightning_model.model.to(device)
    model.eval()
    print(f"✅ Loaded model from {checkpoint_path}")

    # ============================================================
    # LOAD DATA (same as generate_embeddings.py)
    # ============================================================
    data_file = DATA_DIR / "canonical_smiles_subset_10k.csv"
    print(f"\n{'='*70}")
    print("LOADING AND PROCESSING DATA")
    print(f"{'='*70}")
    print(f"Reading: {data_file}")
    
    df = pd.read_csv(data_file)
    print(f"Original dataset: {len(df)} molecules")
    
    # Filter molecules (same as generate_embeddings.py)
    print("\nFiltering molecules...")
    df["valid_smiles"] = df["smiles"].apply(has_max_64_atoms)
    filtered_df = df[df["valid_smiles"]].copy()
    print(f"After filtering: {len(filtered_df)} molecules ({len(df) - len(filtered_df)} removed)")
    
    # Store original indices for alignment
    original_indices = filtered_df.index.to_numpy()
    print(f"✅ Saved original indices for property alignment")
    
    # ============================================================
    # TOKENIZE AND CREATE DATASET (same as generate_embeddings.py)
    # ============================================================
    tokenizer = SMILESTokenizer()
    smiles_list = filtered_df["smiles"].tolist()
    
    print(f"\nTokenizing {len(smiles_list)} SMILES...")
    tokenized = [tokenizer.tokenize(s) for s in tqdm(smiles_list, desc="Tokenizing")]
    
    dataset = MaskedMoleculeDataset(
        tokenized,
        mask_ratio_atoms=0.0,  # No masking for visualization
        mask_ratio_bonds=0.0,
    )

    
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    print(f"✅ Created dataset with {len(dataset)} molecules")

    # ============================================================
    # EXTRACT EMBEDDINGS (same as generate_embeddings.py)
    # ============================================================
    print(f"\n{'='*70}")
    print("EXTRACTING EMBEDDINGS")
    print(f"{'='*70}")
    
    all_embeddings = []

    with torch.no_grad():
        for data in tqdm(dataset, desc="Embedding molecules"):
            data = data.to(device)

        # add fake batch dimension (1 graph)
            batch = Batch.from_data_list([data])
            emb = model.get_embedding(batch)

            all_embeddings.append(emb.cpu().numpy())

        embeddings = np.concatenate(all_embeddings, axis=0)
        
    print(f"✅ Extracted {embeddings.shape[0]} embeddings (dim={embeddings.shape[1]})")
    print("Total SMILES in CSV:", len(df))
    print("Total embeddings generated:", embeddings.shape[0])


    # Save embeddings
    np.save(output_dir / "embeddings.npy", embeddings)

    # ============================================================
    # DIAGNOSE EMBEDDINGS
    # ============================================================
    embeddings_healthy = diagnose_embeddings(embeddings)
    
    if not embeddings_healthy:
        print(f"\n{'='*70}")
        print("⚠️  STOPPING: Fix embedding issues first")
        print(f"{'='*70}")
        print("\n🔧 Check:")
        print("1. Is model.get_embedding() returning the right layer?")
        print("2. Was the model actually trained?")
        print("3. Are model weights loaded correctly?")
        return

    # ============================================================
    # ALIGN PROPERTIES WITH EMBEDDINGS
    # ============================================================
    print(f"\n{'='*70}")
    print("ALIGNING PROPERTIES")
    print(f"{'='*70}")
    
    # Use the filtered dataframe with correct alignment
    props_df = filtered_df[["molecular_weight", "logp", "hba", "hbd"]].reset_index(drop=True)
    
    assert len(props_df) == len(embeddings), \
        f"Mismatch: {len(props_df)} properties vs {len(embeddings)} embeddings"
    
    print(f"✅ Properties perfectly aligned: {len(props_df)} rows")
    
    # Property completeness
    print("\nProperty completeness:")
    properties = ["molecular_weight", "logp", "hba", "hbd"]
    for prop in properties:
        valid = props_df[prop].notna().sum()
        print(f"  {prop}: {valid}/{len(props_df)} ({100*valid/len(props_df):.1f}%)")

    # ============================================================
    # DIMENSIONALITY REDUCTION
    # ============================================================
    print(f"\n{'='*70}")
    print("DIMENSIONALITY REDUCTION: PCA")
    print(f"{'='*70}")
    
    reducer = PCA(n_components=2, random_state=42)
    embeddings_2d = reducer.fit_transform(embeddings)
    print(f"Explained variance: {reducer.explained_variance_ratio_.sum():.2%}")

    # ============================================================
    # VISUALIZATIONS
    # ============================================================
    print(f"\n{'='*70}")
    print("GENERATING VISUALIZATIONS")
    print(f"{'='*70}")
    
    # Plain plot
    plt.figure(figsize=(12, 9))
    scatter = plt.scatter(
        embeddings_2d[:, 0],
        embeddings_2d[:, 1],
        s=20,
        alpha=0.6,
        c=range(len(embeddings_2d)),
        cmap='viridis',
        edgecolors='none'
    )
    plt.colorbar(scatter, label='Molecule index', fraction=0.046, pad=0.04)
    plt.title(f"GIN Molecular Embeddings (PCA)\n{len(embeddings)} molecules")
    plt.xlabel("PCA-1")
    plt.ylabel("PCA-2")
    plt.tight_layout()
    plt.savefig(output_dir / "pca_plain.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Saved plain PCA plot")

    # Property-colored plots
    print("\nGenerating property-colored plots:")
    for prop in properties:
        plot_colored(embeddings_2d, props_df[prop], prop, output_dir, "PCA")

    # ============================================================
    # SAVE METADATA
    # ============================================================
    metadata = {
        "model": "GIN",
        "num_molecules_original": len(df),
        "num_molecules_filtered": len(embeddings),
        "num_filtered_out": len(df) - len(embeddings),
        "embedding_dim": int(embeddings.shape[1]),
        "reduction_method": "PCA",
        "explained_variance": float(reducer.explained_variance_ratio_.sum()),
        "checkpoint": str(checkpoint_path),
        "properties": properties,
        "embeddings_healthy": embeddings_healthy,
    }

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n{'='*70}")
    print(f"✅ ALL VISUALIZATIONS SAVED TO: {output_dir}")
    print(f"{'='*70}")
    print("\nGenerated files:")
    print("  - embeddings.npy")
    print("  - pca_plain.png")
    for prop in properties:
        print(f"  - pca_{prop}.png")
    print("  - metadata.json")
    print(f"\n{'='*70}")


if __name__ == "__main__":
    main()
