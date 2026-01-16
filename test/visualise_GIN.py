import os
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from lightning.data import StreamingDataset
from train.lightning_model_GIN import GraphMoleculeLightningGIN


BASE_DATA_DIR = "data"
BASE_RESULTS_DIR = "results"


def diagnose_embeddings(embeddings):
    """Diagnose embedding quality before visualization."""
    print(f"\n{'='*70}")
    print("EMBEDDING DIAGNOSTICS")
    print(f"{'='*70}")
    
    print(f"Shape: {embeddings.shape}")
    print(f"\nBasic Statistics:")
    print(f"  Mean:   {embeddings.mean():.6f}")
    print(f"  Std:    {embeddings.std():.6f}")
    print(f"  Min:    {embeddings.min():.6f}")
    print(f"  Max:    {embeddings.max():.6f}")
    print(f"  Median: {np.median(embeddings):.6f}")
    
    print(f"\nData Quality:")
    print(f"  Contains NaN: {np.isnan(embeddings).any()}")
    print(f"  Contains Inf: {np.isinf(embeddings).any()}")
    print(f"  All zeros:    {(embeddings == 0).all()}")
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
    print(f"  Min std:  {per_dim_std.min():.6f}")
    print(f"  Max std:  {per_dim_std.max():.6f}")
    print(f"  Dims with std < 0.001: {(per_dim_std < 0.001).sum()}/{len(per_dim_std)}")
    
    # Sample embeddings
    print(f"\nFirst 3 embeddings (first 10 dims):")
    for i in range(min(3, len(embeddings))):
        print(f"  [{i}] {embeddings[i][:10]}")
    
    # Diagnosis
    issues = []
    if embeddings.std() < 0.01:
        issues.append("❌ CRITICAL: Overall std < 0.01 - embeddings have no variation")
    if len(unique_embeddings) < len(embeddings) * 0.1:
        issues.append("❌ CRITICAL: >90% duplicate embeddings - model not learning")
    if (embeddings == 0).sum() / embeddings.size > 0.9:
        issues.append("❌ CRITICAL: >90% values are zero - model might be broken")
    if per_dim_std.mean() < 0.01:
        issues.append("⚠️  WARNING: Very low per-dimension variance")
    
    if issues:
        print(f"\n{'='*70}")
        print("⚠️  ISSUES DETECTED:")
        print(f"{'='*70}")
        for issue in issues:
            print(issue)
        print("\n🔧 RECOMMENDED FIXES:")
        print("1. Verify model was actually trained (check training logs)")
        print("2. Check model.get_embedding() returns the correct layer")
        print("3. Try different embedding layers (before/after pooling)")
        print("4. Ensure model is in eval mode and weights are loaded")
        print("5. Check if you need to normalize embeddings")
        return False
    else:
        print(f"\n✅ Embeddings look healthy!")
        return True


def plot_colored(embeddings_2d, values, name, output_dir, method="PCA"):
    """Plot embeddings colored by property values."""
    values = values.to_numpy()
    valid_mask = ~np.isnan(values)
    
    print(f"\n{name}:")
    print(f"  Valid values: {valid_mask.sum()}/{len(values)} ({100*valid_mask.sum()/len(values):.1f}%)")

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

    plt.title(f"GIN Embeddings ({method}) colored by {name}\n{len(embeddings_2d)} molecules total")
    plt.xlabel(f"{method}-1")
    plt.ylabel(f"{method}-2")
    plt.legend(markerscale=1.5, frameon=True, loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{method.lower()}_{name}.png"), dpi=300, bbox_inches='tight')
    plt.close()


def reduce_dimensions(embeddings, method='pca', **kwargs):
    """
    Reduce embeddings to 2D using specified method.
    
    Args:
        embeddings: numpy array of shape (n_samples, n_features)
        method: 'pca', 'tsne', or 'umap'
        **kwargs: additional parameters for the reduction method
    """
    print(f"\n{'='*70}")
    print(f"DIMENSIONALITY REDUCTION: {method.upper()}")
    print(f"{'='*70}")
    
    if method.lower() == 'pca':
        print("Using PCA (fast, preserves global structure)...")
        reducer = PCA(n_components=2, random_state=42)
        embeddings_2d = reducer.fit_transform(embeddings)
        print(f"Explained variance: {reducer.explained_variance_ratio_.sum():.2%}")
        return embeddings_2d, 'PCA'
        
    elif method.lower() == 'tsne':
        print("Using t-SNE (slower, preserves local structure)...")
        print("This may take several minutes...")
        perplexity = kwargs.get('perplexity', 30)
        reducer = TSNE(
            n_components=2,
            perplexity=min(perplexity, len(embeddings) - 1),
            random_state=42,
            n_iter=1000,
            verbose=1
        )
        embeddings_2d = reducer.fit_transform(embeddings)
        return embeddings_2d, 't-SNE'
        
    elif method.lower() == 'umap':
        print("Using UMAP (balanced speed and quality)...")
        try:
            import umap
            n_neighbors = min(kwargs.get('n_neighbors', 15), len(embeddings) - 1)
            reducer = umap.UMAP(
                n_neighbors=n_neighbors,
                min_dist=kwargs.get('min_dist', 0.1),
                n_components=2,
                metric=kwargs.get('metric', 'euclidean'),
                random_state=42,
                verbose=False
            )
            embeddings_2d = reducer.fit_transform(embeddings)
            return embeddings_2d, 'UMAP'
        except ImportError:
            print("⚠️  UMAP not available, falling back to PCA")
            return reduce_dimensions(embeddings, method='pca')
    else:
        raise ValueError(f"Unknown method: {method}. Use 'pca', 'tsne', or 'umap'")


def load_aligned_properties(props_path, embeddings_count):
    """Load and align properties with graph dataset."""
    props_df = pd.read_csv(props_path)
    
    print(f"\n{'='*70}")
    print("PROPERTY ALIGNMENT")
    print(f"{'='*70}")
    print(f"Original CSV rows: {len(props_df)}")
    print(f"Graph dataset size: {embeddings_count}")
    print(f"Difference: {len(props_df) - embeddings_count} molecules filtered")
    
    # Check for saved index mapping
    index_file = os.path.join(BASE_DATA_DIR, "optimized_graph_dataset", "valid_indices.npy")
    
    if os.path.exists(index_file):
        print(f"\n✅ Found index mapping: {index_file}")
        valid_indices = np.load(index_file)
        if len(valid_indices) == embeddings_count:
            aligned_props = props_df.iloc[valid_indices].reset_index(drop=True)
            print(f"✅ Properties aligned using saved indices")
            return aligned_props
        else:
            print(f"⚠️  Index file has wrong length ({len(valid_indices)} vs {embeddings_count})")
    
    print(f"\n⚠️  No valid index mapping found")
    print("⚠️  Using first N rows (MAY CAUSE MISALIGNMENT)")
    print("⚠️  To fix: Save valid_indices.npy during preprocessing")
    
    aligned_props = props_df.iloc[:embeddings_count].reset_index(drop=True)
    return aligned_props


def main():
    model_name = "GIN"
    output_dir = os.path.join(BASE_RESULTS_DIR, f"plots_{model_name}")
    os.makedirs(output_dir, exist_ok=True)

    print(f"{'='*70}")
    print("GIN MOLECULAR EMBEDDING VISUALIZATION")
    print(f"{'='*70}")

    # ---------------- Load model ----------------
    checkpoint_path = os.path.join(BASE_RESULTS_DIR, "models", "final_model_GIN.ckpt")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    lightning_model = GraphMoleculeLightningGIN.load_from_checkpoint(checkpoint_path)
    model = lightning_model.model
    model.to(device)
    model.eval()
    print(f"✅ Loaded model from {checkpoint_path}")

    # ---------------- Extract embeddings ----------------
    full_dataset = StreamingDataset(
        input_dir=os.path.join(BASE_DATA_DIR, "optimized_graph_dataset")
    )

    print(f"\nExtracting embeddings from {len(full_dataset)} molecules...")
    all_embeddings = []

    with torch.no_grad():
        for data in tqdm(full_dataset, desc="Processing molecules"):
            data = data.to(device)
            data.batch = torch.zeros(data.x.size(0), dtype=torch.long, device=device)
            
            emb = model.get_embedding(data)
            all_embeddings.append(emb.cpu().numpy())

    embeddings = np.concatenate(all_embeddings, axis=0)
    print(f"✅ Extracted {embeddings.shape[0]} embeddings (dim={embeddings.shape[1]})")

    # Save raw embeddings
    np.save(os.path.join(output_dir, "embeddings.npy"), embeddings)

    # ---------------- Diagnose embeddings ----------------
    embeddings_healthy = diagnose_embeddings(embeddings)
    
    if not embeddings_healthy:
        print(f"\n{'='*70}")
        print("⚠️  STOPPING: Fix embedding issues before visualization")
        print(f"{'='*70}")
        return

    # ---------------- Load properties ----------------
    props_path = os.path.join(BASE_DATA_DIR, "canonical_smiles_subset_10k.csv")
    if not os.path.exists(props_path):
        raise FileNotFoundError(f"Properties not found: {props_path}")

    props_df = load_aligned_properties(props_path, len(embeddings))
    
    # Verify alignment
    assert len(props_df) == len(embeddings), \
        f"Mismatch: {len(props_df)} properties vs {len(embeddings)} embeddings"
    
    print(f"\n✅ Properties loaded and aligned: {len(props_df)} rows")
    
    # Property completeness
    print("\nProperty completeness:")
    properties = ["molecular_weight", "logp", "hba", "hbd"]
    for prop in properties:
        if prop in props_df.columns:
            valid = props_df[prop].notna().sum()
            print(f"  {prop}: {valid}/{len(props_df)} ({100*valid/len(props_df):.1f}%)")

    # ---------------- Dimensionality reduction ----------------
    # Try PCA first (always works, fast)
    embeddings_2d, method = reduce_dimensions(embeddings, method='pca')
    
    # ---------------- Plain plot ----------------
    print(f"\nGenerating visualizations...")
    plt.figure(figsize=(12, 9))
    plt.scatter(
        embeddings_2d[:, 0],
        embeddings_2d[:, 1],
        s=20,
        alpha=0.6,
        c=range(len(embeddings_2d)),
        cmap='viridis',
        edgecolors='none'
    )
    plt.colorbar(label='Molecule index', fraction=0.046, pad=0.04)
    plt.title(f"GIN Molecular Embeddings ({method})\n{len(embeddings)} molecules")
    plt.xlabel(f"{method}-1")
    plt.ylabel(f"{method}-2")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{method.lower()}_plain.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved plain {method} plot")

    # ---------------- Property-colored plots ----------------
    for prop in properties:
        if prop in props_df.columns:
            plot_colored(embeddings_2d, props_df[prop], prop, output_dir, method)

    # ---------------- Metadata ----------------
    metadata = {
        "model": model_name,
        "num_molecules": int(embeddings.shape[0]),
        "embedding_dim": int(embeddings.shape[1]),
        "reduction_method": method,
        "checkpoint": checkpoint_path,
        "properties": properties,
        "embeddings_healthy": embeddings_healthy,
    }

    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n{'='*70}")
    print(f"✅ ALL VISUALIZATIONS SAVED TO: {output_dir}")
    print(f"{'='*70}")
    print("\nGenerated files:")
    print(f"  - embeddings.npy")
    print(f"  - {method.lower()}_plain.png")
    for prop in properties:
        if prop in props_df.columns:
            print(f"  - {method.lower()}_{prop}.png")
    print(f"  - metadata.json")


if __name__ == "__main__":
    main()