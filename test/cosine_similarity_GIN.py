import sys
from pathlib import Path
import json

import torch
from sklearn.metrics.pairwise import cosine_similarity

# =====================================================
# Fix Python path
# =====================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from preprocess.tokenizer import SMILESTokenizer
from train.lightning_model_GIN import GraphMoleculeLightningGIN
from train.utils import create_masked_graph_from_tensors

# =====================================================
# Paths
# =====================================================
OUTPUT_DIR = Path("results/plots_GIN")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = OUTPUT_DIR / "cosine_similarity_GIN.json"

# =====================================================
# Molecule pairs
# =====================================================
MOLECULE_PAIRS = {
    "homologous": ("Cc1ccccc1", "CCc1ccccc1"),
    "isomers": ("CCCO", "CC(C)O"),
    "dissimilar": ("CC(=O)OC1=CC=CC=C1C(=O)O", "C(C1C(C(C(C(O1)O)O)O)O)O"),
    "stereoisomers": ("C[C@H](N)C(=O)O", "C[C@@H](N)C(=O)O"),
    "bioisosteres": ("c1ccc(C(=O)O)cc1", "c1cnc(C(=O)O)cc1"),
    "functional_isomers": ("CCO", "COC"),
}

EXPECTED_RANGES = {
    "homologous": (0.95, 1.00),
    "isomers": (0.90, 0.98),
    "dissimilar": (0.00, 0.40),
    "stereoisomers": (0.95, 1.00),
    "bioisosteres": (0.75, 0.90),
    "functional_isomers": (0.40, 0.65),
}


def get_embedding(smiles, model, tokenizer, device):
    tokenized = tokenizer.tokenize(smiles)
    data = create_masked_graph_from_tensors(
        tokenized["atomic_numbers"],
        tokenized["bond_matrix"],
        mask_ratio_atoms=0.0,
        mask_ratio_bonds=0.0,
    )
    data = data.to(device)
    with torch.no_grad():
        emb = model.get_graph_embedding(data)
    return emb.cpu().numpy()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = SMILESTokenizer()
    lightning_model = GraphMoleculeLightningGIN(
        hidden_dim=128,
        num_layers=3,
    ).to(device)

    model = lightning_model.model
    model.eval()

    results = {}

    for key, (smi1, smi2) in MOLECULE_PAIRS.items():
        emb1 = get_embedding(smi1, model, tokenizer, device)
        emb2 = get_embedding(smi2, model, tokenizer, device)

        sim = cosine_similarity(emb1, emb2)[0, 0]
        lo, hi = EXPECTED_RANGES[key]

        results[key] = {
            "pair": [smi1, smi2],
            "cosine_similarity": float(sim),
            "expected_min": lo,
            "expected_max": hi,
            "within_expected_range": bool(lo <= sim <= hi),
        }

    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2)

    print(f"✅ GIN cosine similarity saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
