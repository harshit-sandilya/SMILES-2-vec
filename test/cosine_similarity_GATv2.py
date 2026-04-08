import sys
from pathlib import Path
import argparse
import json
import torch
from sklearn.metrics.pairwise import cosine_similarity

# =====================================================
# Fix Python path
# =====================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from preprocess.tokenize import SMILESTokenizer
from train.lightning_model_GATv2 import GraphMoleculeLightningGATv2
from train.utils import get_single_embedding

# =====================================================
# Args
# =====================================================
parser = argparse.ArgumentParser()
parser.add_argument("--model-file", type=str, default="results/models/final_model_gatv2.ckpt")
args = parser.parse_args()

model_file = (
    args.model_file
    if Path(args.model_file).is_absolute()
    else str(Path("results/models") / args.model_file)
)

# =====================================================
# Paths
# =====================================================
OUTPUT_DIR = Path("results/cosine_similarity_gatv2_zinc")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = OUTPUT_DIR / "cosine_similarity.json"

# =====================================================
# Molecule pairs
# =====================================================
MOLECULE_PAIRS = {
    "homologous":         ("Cc1ccccc1",                    "CCc1ccccc1"),
    "isomers":            ("CCCO",                         "CC(C)O"),
    "dissimilar":         ("CC(=O)OC1=CC=CC=C1C(=O)O",    "C(C1C(C(C(C(O1)O)O)O)O)O"),
    "stereoisomers":      ("C[C@H](N)C(=O)O",             "C[C@@H](N)C(=O)O"),
    "bioisosteres":       ("c1ccc(C(=O)O)cc1",            "c1cnc(C(=O)O)cc1"),
    "functional_isomers": ("CCO",                         "COC"),
}

EXPECTED_RANGES = {
    "homologous":         (0.95, 1.00),
    "isomers":            (0.90, 0.98),
    "dissimilar":         (0.00, 0.40),
    "stereoisomers":      (0.95, 1.00),
    "bioisosteres":       (0.75, 0.90),
    "functional_isomers": (0.40, 0.65),
}

# =====================================================
# Load model (manual — avoids architecture mismatch)
# =====================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
print(f"Loading checkpoint: {model_file}")

raw_ckpt = torch.load(model_file, map_location=device, weights_only=False)
hparams  = raw_ckpt.get("hyper_parameters", {})
print(f"Checkpoint hparams: {hparams}")

lightning_model = GraphMoleculeLightningGATv2(
    hidden_dim              = hparams.get("hidden_dim",              512),
    num_layers              = hparams.get("num_layers",                6),
    num_heads               = hparams.get("num_heads",                 8),
    lr                      = hparams.get("lr",                     2e-4),
    weight_decay            = hparams.get("weight_decay",           1e-5),
    use_contrastive         = hparams.get("use_contrastive",        False),
    contrastive_weight      = hparams.get("contrastive_weight",      0.1),
    contrastive_temperature = hparams.get("contrastive_temperature", 0.07),
    warmup_steps            = hparams.get("warmup_steps",           2000),
)

missing, unexpected = lightning_model.load_state_dict(
    raw_ckpt["state_dict"], strict=False
)
if missing:
    print(f"[warn] Missing keys: {len(missing)}")
if unexpected:
    print(f"[warn] Unexpected keys: {len(unexpected)}")

lightning_model.eval()
model = lightning_model.model.to(device)
model.eval()
print("Model loaded successfully.")

tokenizer = SMILESTokenizer()

# =====================================================
# Compute similarities
# =====================================================
results = {}

for key, (smi1, smi2) in MOLECULE_PAIRS.items():
    emb1 = get_single_embedding(smi1, model, tokenizer, device)
    emb2 = get_single_embedding(smi2, model, tokenizer, device)

    if emb1 is None or emb2 is None:
        print(f"{key:20s}: skipped (embedding failed)")
        continue

    emb1 = emb1.reshape(1, -1)
    emb2 = emb2.reshape(1, -1)

    sim  = cosine_similarity(emb1, emb2)[0, 0]
    lo, hi = EXPECTED_RANGES[key]

    results[key] = {
        "pair":                  [smi1, smi2],
        "cosine_similarity":     float(sim),
        "expected_min":          lo,
        "expected_max":          hi,
        "within_expected_range": bool(lo <= sim <= hi),
    }

    print(
        f"{key:20s}: {sim:.4f} | expected [{lo}, {hi}] "
        f"=> {'OK' if lo <= sim <= hi else 'FAIL'}"
    )

# =====================================================
# Save
# =====================================================
with open(OUTPUT_FILE, "w") as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved to: {OUTPUT_FILE}")