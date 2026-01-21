import sys
from pathlib import Path

# =====================================================
# Fix Python path (IMPORTANT)
# =====================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from argparse import ArgumentParser
import os
import json

import torch
from sklearn.metrics.pairwise import cosine_similarity

from preprocess.tokenizer import SMILESTokenizer
from train.lightning_model import GraphMoleculeLightning
from train.utils import get_single_embedding

# =====================================================
# Paths
# =====================================================
BASE_RESULTS_DIR = "results"
OUTPUT_DIR = os.path.join(BASE_RESULTS_DIR, "plots_GATv2")
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_FILE = os.path.join(OUTPUT_DIR, "cosine_similarity_GATv2.json")

# =====================================================
# Args
# =====================================================
parser = ArgumentParser()
parser.add_argument(
    "--model-file",
    type=str,
    default="final_model.ckpt",
    help="Model checkpoint filename (loaded from results/models/)",
)
args = parser.parse_args()

model_file = (
    args.model_file
    if os.path.isabs(args.model_file)
    else os.path.join("results", "models", args.model_file)
)

# =====================================================
# Load model
# =====================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

lightning_model = GraphMoleculeLightning.load_from_checkpoint(
    model_file,
    strict=False,
)
inference_model = lightning_model.model.to(device)
inference_model.eval()

print("✅ GATv2 model loaded successfully.")

tokenizer = SMILESTokenizer()

# =====================================================
# Molecule pairs
# =====================================================
MOLECULE_PAIRS = {
    "homologous": (
        "Cc1ccccc1",      # toluene
        "CCc1ccccc1",     # ethylbenzene
    ),
    "isomers": (
        "CCCO",           # propanol
        "CC(C)O",         # isopropanol
    ),
    "dissimilar": (
        "CC(=O)OC1=CC=CC=C1C(=O)O",  # aspirin
        "C(C1C(C(C(C(O1)O)O)O)O)O",  # glucose
    ),
    "stereoisomers": (
        "C[C@H](N)C(=O)O",   # R-alanine
        "C[C@@H](N)C(=O)O",  # S-alanine
    ),
    "bioisosteres": (
        "c1ccc(C(=O)O)cc1",  # benzoic acid
        "c1cnc(C(=O)O)cc1",  # nicotinic acid
    ),
    "functional_isomers": (
        "CCO",  # ethanol
        "COC",  # dimethyl ether
    ),
}

# =====================================================
# Compute similarities
# =====================================================
results = {}

for name, (smi1, smi2) in MOLECULE_PAIRS.items():
    emb1 = get_single_embedding(smi1, inference_model, tokenizer, device)
    emb2 = get_single_embedding(smi2, inference_model, tokenizer, device)

    sim = cosine_similarity(emb1, emb2)[0, 0]

    results[name] = {
        "pair": [smi1, smi2],
        "cosine_similarity": float(sim),
    }

    print(f"{name:20s}: {sim:.4f}")

# =====================================================
# Save results
# =====================================================
with open(OUTPUT_FILE, "w") as f:
    json.dump(results, f, indent=2)

print(f"\n✅ GATv2 cosine similarity results saved to:")
print(f"   {OUTPUT_FILE}")
