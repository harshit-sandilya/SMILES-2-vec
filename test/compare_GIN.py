import sys
from pathlib import Path

# =====================================================
# Fix Python path
# =====================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import os
import json
import warnings
import torch
import numpy as np
import deepchem as dc

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, MACCSkeys, rdMolDescriptors
from rdkit.DataStructs import cDataStructs

from torch_geometric.loader import DataLoader
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from config import *
from preprocess.dataset import MaskedMoleculeDataset
from preprocess.tokenizer import SMILESTokenizer
from train.lightning_model_GIN import GraphMoleculeLightningGIN

# =====================================================
# Setup
# =====================================================
RDLogger.DisableLog("rdApp.WARNING")
warnings.filterwarnings("ignore")

RESULT_DIR = "results/plots_GIN"
os.makedirs(RESULT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =====================================================
# Load GIN encoder (Lightning-safe)
# =====================================================
model = GraphMoleculeLightningGIN(
    hidden_dim=hidden_dim,
    num_layers=num_layers,
)

state = torch.load(
    "results/models/final_model_GIN.ckpt",
    map_location=device,
)

# Handle Lightning vs raw state_dict
if "state_dict" in state:
    state = {
        k.replace("model.", ""): v
        for k, v in state["state_dict"].items()
        if k.startswith("model.")
    }

model.model.load_state_dict(state, strict=False)

model.to(device)
model.eval()

tokenizer = SMILESTokenizer()

print("✅ GIN encoder loaded")

# =====================================================
# RDKit helpers
# =====================================================
def _fp_to_np(fp, n_bits):
    arr = np.zeros(n_bits, dtype=np.int8)
    cDataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def generate_ecfp(m):
    return _fp_to_np(AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048), 2048)


def generate_maccs(m):
    return _fp_to_np(MACCSkeys.GenMACCSKeys(m), 166)


def generate_rdkit_fp(m):
    return _fp_to_np(Chem.RDKFingerprint(m, fpSize=2048), 2048)


def generate_atom_pairs(m):
    return _fp_to_np(
        rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(m, 2048), 2048
    )


# =====================================================
# Evaluation
# =====================================================
def run_evaluation(X_train, y_train, X_test, y_test, task_type):
    if task_type == "regression":
        model = XGBRegressor(
            objective="reg:squarederror",
            n_jobs=-1,
            random_state=42,
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        return {
            "R2": r2_score(y_test, preds),
            "RMSE": np.sqrt(mean_squared_error(y_test, preds)),
        }

    else:
        if len(np.unique(y_test)) < 2:
            return {}

        model = XGBClassifier(
            eval_metric="logloss",
            n_jobs=-1,
            random_state=42,
        )
        model.fit(X_train, y_train)
        probs = model.predict_proba(X_test)[:, 1]
        preds = (probs > 0.5).astype(int)
        return {
            "AUC": roc_auc_score(y_test, probs),
            "Accuracy": accuracy_score(y_test, preds),
            "Precision": precision_score(y_test, preds, zero_division=0),
            "Recall": recall_score(y_test, preds, zero_division=0),
            "F1": f1_score(y_test, preds, zero_division=0),
        }


# =====================================================
# Featurization (FIXED – identical logic to GCN)
# =====================================================
def featurize(dc_dataset):
    df = dc_dataset.to_dataframe()

    # ---- Normalize X → SMILES STRING ----
    def to_smiles(x):
        if isinstance(x, Chem.Mol):
            return Chem.MolToSmiles(x)
        if isinstance(x, str):
            return x
        return None

    df["smiles"] = df["X"].apply(to_smiles)
    df = df.dropna(subset=["smiles"])

    # ---- Build RDKit mols safely ----
    df["mol"] = df["smiles"].apply(Chem.MolFromSmiles)
    df = df.dropna(subset=["mol"])

    # ---- Atom count filter ----
    df["num_atoms"] = df["mol"].apply(lambda m: m.GetNumAtoms())
    df = df[df["num_atoms"] <= 64].reset_index(drop=True)

    if df.empty:
        return None, None

    y = df.filter(regex="^y").values

    # ---- Tokenize safely ----
    df["tokenized"] = df["smiles"].apply(tokenizer.tokenize)

    dataset = MaskedMoleculeDataset(
        df["tokenized"].tolist(),
        mask_ratio_atoms=0.0,
        mask_ratio_bonds=0.0,
    )

    loader = DataLoader(dataset, batch_size=256, shuffle=False)

    with torch.no_grad():
        gin_emb = np.concatenate(
            [
                model.model.get_graph_embedding(batch.to(device)).cpu().numpy()
                for batch in loader
            ],
            axis=0,
        )

    X = {
        "GIN": gin_emb,
        "ECFP": np.stack(df["mol"].apply(generate_ecfp)),
        "MACCS": np.stack(df["mol"].apply(generate_maccs)),
        "RDKit": np.stack(df["mol"].apply(generate_rdkit_fp)),
        "AtomPairs": np.stack(df["mol"].apply(generate_atom_pairs)),
    }

    return y, X


# =====================================================
# Benchmarks
# =====================================================
BENCHMARKS = [
    ("ESOL", dc.molnet.load_delaney, "regression"),
    ("Lipophilicity", dc.molnet.load_lipo, "regression"),
    ("BACE", dc.molnet.load_bace_classification, "classification"),
    ("SIDER", dc.molnet.load_sider, "classification"),
]

all_results = []

for name, loader_fn, task_type in BENCHMARKS:
    print(f"\n▶ {name}")

    try:
        _, (train_set, _, test_set), _ = loader_fn(
            featurizer="Raw",
            splitter="scaffold",
            reload=True,
        )
    except Exception as e:
        print(f"⚠️ Skipping {name}: {e}")
        continue

    y_train, X_train = featurize(train_set)
    y_test, X_test = featurize(test_set)

    if y_train is None or y_test is None:
        print(f"⚠️ No valid molecules for {name}")
        continue

    result_entry = {
        "dataset": name,
        "task": task_type,
        "metrics": {},
    }

    for method in X_train:
        metrics = run_evaluation(
            X_train[method],
            y_train[:, 0],
            X_test[method],
            y_test[:, 0],
            task_type,
        )
        if metrics:
            result_entry["metrics"][method] = metrics

    all_results.append(result_entry)

# =====================================================
# Save
# =====================================================
output_file = os.path.join(RESULT_DIR, "benchmark_results.json")
with open(output_file, "w") as f:
    json.dump(all_results, f, indent=2)

print(f"\n✅ GIN benchmark results saved to {output_file}")
