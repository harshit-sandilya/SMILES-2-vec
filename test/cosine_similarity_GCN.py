import sys
from pathlib import Path
import json
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from rdkit import Chem
from rdkit.DataStructs import TanimotoSimilarity
from rdkit.Chem import AllChem

# Fix Python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from preprocess.tokenizer import SMILESTokenizer
from train.lightning_model_GCN import GraphMoleculeLightningGCN
from train.utils import create_masked_graph_from_tensors

# Paths
OUTPUT_DIR = Path("results/plots_GCN")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = OUTPUT_DIR / "cosine_similarity_GCN.json"
CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints/best_model.ckpt" # Update this path!

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
    "isomers": (0.70, 0.90), # Adjusted: Isomers shouldn't be 0.99
    "dissimilar": (0.00, 0.40),
    "stereoisomers": (0.95, 1.00),
    "bioisosteres": (0.75, 0.90),
    "functional_isomers": (0.30, 0.60),
}

def get_tanimoto(smi1, smi2):
    """Calculates Tanimoto similarity as a chemical baseline."""
    m1, m2 = Chem.MolFromSmiles(smi1), Chem.MolFromSmiles(smi2)
    fp1 = AllChem.GetMorganFingerprintAsBitVect(m1, 2, nBits=2048)
    fp2 = AllChem.GetMorganFingerprintAsBitVect(m2, 2, nBits=2048)
    return TanimotoSimilarity(fp1, fp2)

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
        # Normalize the embedding to unit length for accurate cosine similarity
        emb = torch.nn.functional.normalize(emb, p=2, dim=1)
    return emb.cpu().numpy()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = SMILESTokenizer()

    # LOAD TRAINED WEIGHTS
    if CHECKPOINT_PATH.exists():
        print(f"Loading checkpoint from {CHECKPOINT_PATH}...")
        lightning_model = GraphMoleculeLightningGCN.load_from_checkpoint(
            CHECKPOINT_PATH, 
            hidden_dim=128, 
            num_layers=3
        )
    else:
        print("⚠️ No checkpoint found! Running with random weights.")
        lightning_model = GraphMoleculeLightningGCN(hidden_dim=128, num_layers=3)

    model = lightning_model.model.to(device)
    model.eval()

    results = {}

    for key, (smi1, smi2) in MOLECULE_PAIRS.items():
        emb1 = get_embedding(smi1, model, tokenizer, device)
        emb2 = get_embedding(smi2, model, tokenizer, device)

        sim = cosine_similarity(emb1, emb2)[0, 0]
        tanimoto = get_tanimoto(smi1, smi2)
        lo, hi = EXPECTED_RANGES[key]

        results[key] = {
            "pair": [smi1, smi2],
            "cosine_similarity": round(float(sim), 4),
            "tanimoto_baseline": round(float(tanimoto), 4),
            "expected_min": lo,
            "expected_max": hi,
            "within_expected_range": bool(lo <= sim <= hi),
        }

    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2)

    print(f"✅ GCN similarity evaluation saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()