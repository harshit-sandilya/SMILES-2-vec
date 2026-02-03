import sys
from pathlib import Path
import json
import glob
import torch
from rdkit import Chem
from sklearn.metrics.pairwise import cosine_similarity

# =====================================================
# Fix Python path
# =====================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from preprocess.preprocess_DMPNN import create_dmpnn_graph
from train.lightning_model_DMPNN import GraphMoleculeLightningDMPNN

# =====================================================
# Paths
# =====================================================
OUTPUT_DIR = Path("results/plots_DMPNN")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = OUTPUT_DIR / "cosine_similarity_DMPNN.json"

CHECKPOINT_DIR = PROJECT_ROOT / "results/checkpoints_DMPNN"

MOLECULE_PAIRS = {
    "homologous": ("Cc1ccccc1", "CCc1ccccc1"),
    "isomers": ("CCCO", "CC(C)O"),
    "dissimilar": (
        "CC(=O)OC1=CC=CC=C1C(=O)O",
        "C(C1C(C(C(C(O1)O)O)O)O)O"
    ),
    "stereoisomers": ("C[C@H](N)C(=O)O", "C[C@@H](N)C(=O)O"),
    "bioisosteres": ("c1ccc(C(=O)O)cc1", "c1cnc(C(=O)O)cc1"),
    "functional_isomers": ("CCO", "COC"),
}

# =====================================================
# Utils
# =====================================================

def get_best_checkpoint():
    ckpts = glob.glob(str(CHECKPOINT_DIR / "*.ckpt"))
    if len(ckpts) == 0:
        return None

    def extract_loss(path):
        name = Path(path).stem
        return float(name.split("val_loss=")[-1])

    return min(ckpts, key=extract_loss)


def smiles_to_graph_inputs(smiles):
    mol = Chem.MolFromSmiles(smiles)

    atomic_numbers = []
    for atom in mol.GetAtoms():
        atomic_numbers.append(atom.GetAtomicNum())

    bond_index = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bond_index.append([i, j])
        bond_index.append([j, i])  # directed

    bond_index = torch.tensor(bond_index, dtype=torch.long).t()
    return atomic_numbers, bond_index



def get_embedding(smiles, model, device):
    atomic_numbers, bond_index = smiles_to_graph_inputs(smiles)
    data = create_dmpnn_graph(atomic_numbers, bond_index)
    data = data.to(device)

    with torch.no_grad():
        emb = model.get_graph_embedding(data)
        emb = torch.nn.functional.normalize(emb, p=2, dim=1)

    return emb.cpu().numpy()


# =====================================================
# Main
# =====================================================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    best_ckpt = get_best_checkpoint()
    if best_ckpt is None:
        print("❌ No checkpoints found.")
        return

    print(f"✅ Loading best checkpoint: {best_ckpt}")

    lightning_model = GraphMoleculeLightningDMPNN.load_from_checkpoint(best_ckpt)
    model = lightning_model.model.to(device)
    model.eval()

    results = {}

    for key, (smi1, smi2) in MOLECULE_PAIRS.items():
        emb1 = get_embedding(smi1, model, device)
        emb2 = get_embedding(smi2, model, device)

        sim = cosine_similarity(emb1, emb2)[0, 0]

        results[key] = {
            "pair": [smi1, smi2],
            "cosine_similarity": float(sim),
        }

    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2)

    print(f"📊 Results saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
