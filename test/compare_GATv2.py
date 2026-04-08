import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import os
import warnings
import json
from argparse import ArgumentParser

import deepchem as dc
import numpy as np
import pandas as pd
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, MACCSkeys
from rdkit.DataStructs import cDataStructs

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from torch_geometric.loader import DataLoader
from xgboost import XGBClassifier, XGBRegressor

from preprocess.tokenize import SMILESTokenizer
from train.lightning_model_GATv2 import GraphMoleculeLightningGATv2
from train.utils import create_masked_graph_from_tensors

BASE_RESULTS_DIR = "results"
os.makedirs(BASE_RESULTS_DIR, exist_ok=True)

RDLogger.DisableLog("rdApp.WARNING")
warnings.filterwarnings("ignore", category=UserWarning)

# =====================================================
# Args
# =====================================================
parser = ArgumentParser()
parser.add_argument("--model-file", type=str, default="gatv2_zinc.ckpt")
args = parser.parse_args()

model_file = (
    args.model_file
    if os.path.isabs(args.model_file)
    else os.path.join("results", "models", args.model_file)
)

print(f"Loading checkpoint: {model_file}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# =====================================================
# Manual checkpoint loader
#
# Why not load_from_checkpoint():
#   Lightning reconstructs the model from hparams saved
#   inside the checkpoint BEFORE loading weights. If the
#   saved hparams produce a different architecture than the
#   current model_GATv2.py (e.g. old checkpoint had
#   atom_embedder [120,64], current code has [120,768])
#   it raises RuntimeError even with strict=False because
#   the size mismatch happens at construction, not loading.
#
# Fix: torch.load the raw checkpoint, read hparams from it,
#   instantiate the LightningModule manually with those hparams,
#   then call load_state_dict(strict=False) to load all weights
#   that match and skip any whose shape changed between versions.
# =====================================================
raw_ckpt = torch.load(model_file, map_location=device, weights_only=False)

hparams = raw_ckpt.get("hyper_parameters", {})
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
    print(f"[warn] Missing keys (random init): {len(missing)} keys")
    for k in missing[:5]:
        print(f"       {k}")
if unexpected:
    print(f"[warn] Unexpected keys (ignored): {len(unexpected)} keys")

lightning_model.eval()
inference_model = lightning_model.model.to(device)
inference_model.eval()
print("Model loaded successfully.")

tokenizer = SMILESTokenizer(max_atoms=256)

# =====================================================
# Dataset
# =====================================================
class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, tokenized_list):
        self.data = tokenized_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        t = self.data[idx]
        return create_masked_graph_from_tensors(
            torch.tensor(t["atomic_numbers"], dtype=torch.long),
            torch.tensor(t["bond_matrix"],    dtype=torch.long),
            0.0,          # mask_ratio_atoms
            0.0,          # mask_ratio_bonds
            t["smiles"],  # smiles
            False,        # apply_masking
        )

# =====================================================
# Featurizers
# =====================================================
def _fp_to_np(fp, n_bits):
    arr = np.zeros(n_bits, dtype=np.int8)
    cDataStructs.ConvertToNumpyArray(fp, arr)
    return arr

def generate_ecfp(mol):
    return _fp_to_np(
        AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048), 2048
    )

def generate_maccs(mol):
    return _fp_to_np(MACCSkeys.GenMACCSKeys(mol), 166)

# =====================================================
# Evaluation
# =====================================================
def run_evaluation(X_train, y_train, X_test, y_test, task_type):
    if len(X_train) == 0 or len(X_test) == 0:
        return {}

    if task_type == "regression":
        model = XGBRegressor(n_jobs=-1, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        return {
            "R2":   round(float(r2_score(y_test, y_pred)), 4),
            "RMSE": round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 4),
        }

    y_train = (y_train > 0).astype(int)
    y_test  = (y_test  > 0).astype(int)

    if len(np.unique(y_test)) < 2:
        print("  Skipping -- only one class in test set")
        return {}

    model = XGBClassifier(n_jobs=-1, eval_metric="logloss", random_state=42)
    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob > 0.5).astype(int)

    return {
        "AUC":      round(float(roc_auc_score(y_test, y_prob)), 4),
        "Accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "F1":       round(float(f1_score(y_test, y_pred)), 4),
    }

# =====================================================
# Manual datasets
# =====================================================
def load_manual_dataset(name):
    base = "data/benchmarks"

    if name == "FreeSolv":
        df = pd.read_csv(f"{base}/SAMPL.csv")
        return df["smiles"], df["expt"], "regression"

    if name == "BBBP":
        df = pd.read_csv(f"{base}/BBBP.csv")
        return df["smiles"], df["p_np"], "classification"

    if name == "ClinTox":
        df = pd.read_csv(f"{base}/clintox.csv.gz")
        df["y"] = df["FDA_APPROVED"]
        return df["smiles"], df["y"], "classification"

    return None, None, None

# =====================================================
# Featurization
# =====================================================
def process_and_featurize_split(data, max_atoms=256):
    if isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        df = pd.DataFrame({
            "smiles": [
                Chem.MolToSmiles(m) if isinstance(m, Chem.Mol) else str(m)
                for m in data.X
            ],
            "y": data.y[:, 0] if data.y.ndim == 2 else data.y,
        })

    if "smiles" not in df.columns:
        for col in df.columns:
            if df[col].dtype == object:
                df = df.rename(columns={col: "smiles"})
                break

    if "y" not in df.columns:
        df["y"] = df.iloc[:, -1]

    df = df[["smiles", "y"]].dropna()
    df["mol"] = df["smiles"].apply(Chem.MolFromSmiles)
    df = df.dropna(subset=["mol"])
    df = df[df["mol"].apply(lambda m: m.GetNumAtoms()) <= max_atoms].reset_index(drop=True)

    print(f"  Molecules after filtering: {len(df)}")

    if df.empty:
        return None, None

    mols = df["mol"].tolist()

    valid_indices = []
    tokenized     = []
    for i, smi in enumerate(df["smiles"]):
        try:
            tok = tokenizer.tokenize(smi)
            tok["smiles"] = smi          # ← store smiles explicitly
            tokenized.append(tok)
            valid_indices.append(i)
        except Exception as e:
            print(f"  Skipping '{smi[:40]}': {e}")

    if not tokenized:
        return None, None

    mols  = [mols[i]  for i in valid_indices]
    y_arr = df["y"].values[valid_indices]

    batch_size = 64 if torch.cuda.is_available() else 16
    loader = DataLoader(SimpleDataset(tokenized), batch_size=batch_size, shuffle=False)

    gnn_feats = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            emb = inference_model.get_embedding(batch)
            gnn_feats.append(emb.cpu().numpy())

    gnn_feats = np.concatenate(gnn_feats, axis=0)

    feats = {
        "GNN":   gnn_feats,
        "ECFP":  np.array([generate_ecfp(m)  for m in mols]),
        "MACCS": np.array([generate_maccs(m) for m in mols]),
    }

    return y_arr, feats

# =====================================================
# Benchmarks
# =====================================================
BENCHMARKS = [
    ("ESOL",          dc.molnet.load_delaney,             "regression"),
    ("Lipophilicity", dc.molnet.load_lipo,                "regression"),
    ("FreeSolv",      None,                               "regression"),
    ("BACE",          dc.molnet.load_bace_classification, "classification"),
    ("BBBP",          None,                               "classification"),
    ("ClinTox",       None,                               "classification"),
    ("SIDER",         dc.molnet.load_sider,               "classification"),
]

results = []

for name, loader_fn, task in BENCHMARKS:
    print(f"\n{'='*50}")
    print(f"  {name}  ({task})")
    print(f"{'='*50}")

    if loader_fn is None:
        smiles, targets, task = load_manual_dataset(name)
        if smiles is None:
            print(f"  Skipping -- file not found in data/benchmarks/")
            continue

        df = pd.DataFrame({"smiles": smiles, "y": targets}).dropna()
        train_df = df.sample(frac=0.8, random_state=42)
        test_df  = df.drop(train_df.index)
        print(f"  Train: {len(train_df)}  Test: {len(test_df)}")

        y_train, X_train = process_and_featurize_split(train_df)
        y_test,  X_test  = process_and_featurize_split(test_df)

    else:
        try:
            _, splits, _ = loader_fn(
                featurizer="Raw", splitter="scaffold", reload=False
            )
            train_dc, _, test_dc = splits
        except Exception as e:
            print(f"  Skipping -- DeepChem load failed: {e}")
            continue

        print(f"  Train: {len(train_dc)}  Test: {len(test_dc)}")
        y_train, X_train = process_and_featurize_split(train_dc)
        y_test,  X_test  = process_and_featurize_split(test_dc)

    if y_train is None or y_test is None:
        print("  Skipping -- featurization returned empty split")
        continue

    for method in X_train:
        metrics = run_evaluation(
            X_train[method], y_train,
            X_test[method],  y_test,
            task,
        )
        print(f"  {method:6s}  {metrics}")
        results.append({"dataset": name, "method": method, "metrics": metrics})

# =====================================================
# Save
# =====================================================
ckpt_stem   = Path(model_file).stem
output_path = f"results/benchmark_results_{ckpt_stem}.json"

with open(output_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"\nDone -- results saved to {output_path}")