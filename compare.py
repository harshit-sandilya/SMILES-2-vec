import os
import warnings
import json
from argparse import ArgumentParser

import deepchem as dc
import numpy as np
import pandas as pd
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors, MACCSkeys, rdMolDescriptors
from rdkit.DataStructs import cDataStructs
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from torch_geometric.loader import DataLoader
from xgboost import XGBClassifier, XGBRegressor

from config import *
from dataset import MaskedMoleculeDataset
from lightning_model import GraphMoleculeLightning
from tokenizer import SMILESTokenizer

RDLogger.DisableLog("rdApp.WARNING")
warnings.filterwarnings("ignore", category=UserWarning)

if not os.path.exists("results"):
    os.makedirs("results")

parser = ArgumentParser()
parser.add_argument(
    "--model-file",
    type=str,
    default="model/final_model.ckpt",
    help="Path to the final model file",
)
args = parser.parse_args()

model_file = args.model_file
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lightning_model = GraphMoleculeLightning.load_from_checkpoint(model_file)
inference_model = lightning_model.model
inference_model.to(device)
inference_model.eval()
tokenizer = SMILESTokenizer()


# ===================== Featurizers =====================
def _fp_to_np(fp, n_bits):
    arr = np.zeros(n_bits, dtype=np.int8)
    cDataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def generate_ecfp(mol, radius=2, n_bits=2048):
    return (
        _fp_to_np(
            AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits), n_bits
        )
        if mol
        else None
    )


def generate_maccs(mol):
    return _fp_to_np(MACCSkeys.GenMACCSKeys(mol), 166) if mol else None


def generate_rdkit_fp(mol, n_bits=2048):
    return _fp_to_np(Chem.RDKFingerprint(mol, fpSize=n_bits), n_bits) if mol else None


def generate_atom_pairs(mol, n_bits=2048):
    return (
        _fp_to_np(
            rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(mol, nBits=n_bits),
            n_bits,
        )
        if mol
        else None
    )


def generate_rdkit_descriptors(mol):
    try:
        descs = Descriptors.CalcMolDescriptors(mol)
        return np.array(
            [
                d if isinstance(d, (int, float)) and np.isfinite(d) else np.nan
                for d in descs
            ],
            dtype=np.float64,
        )
    except:
        return np.full(len(Descriptors._descList), np.nan, dtype=np.float64)


# ===================== Evaluation =====================
def run_evaluation(X_train, y_train, X_test, y_test, task_type, method_name):
    if method_name == "2D Descriptors":
        train_nan_mask = (np.isnan(X_train).sum(axis=1) / X_train.shape[1]) < 0.2
        test_nan_mask = (np.isnan(X_test).sum(axis=1) / X_test.shape[1]) < 0.2
        X_train, y_train = X_train[train_nan_mask], y_train[train_nan_mask]
        X_test, y_test = X_test[test_nan_mask], y_test[test_nan_mask]

    if len(X_train) == 0 or len(X_test) == 0:
        return {}

    if task_type == "regression":
        model = XGBRegressor(random_state=42, n_jobs=-1, objective="reg:squarederror")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        return {
            "R2": r2_score(y_test, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        }
    else:
        if len(np.unique(y_test)) < 2:
            return {}
        scale_pos_weight = (
            (np.sum(y_train == 0) / np.sum(y_train == 1))
            if np.sum(y_train == 1) > 0
            else 1
        )
        model = XGBClassifier(
            random_state=42,
            n_jobs=-1,
            use_label_encoder=False,
            eval_metric="logloss",
            scale_pos_weight=scale_pos_weight,
        )
        model.fit(X_train, y_train)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba > 0.5).astype(int)
        return {
            "AUC": roc_auc_score(y_test, y_pred_proba),
            "Accuracy": accuracy_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred, zero_division=0),
            "Precision": precision_score(y_test, y_pred, zero_division=0),
            "F1": f1_score(y_test, y_pred, zero_division=0),
        }


# ===================== Featurize + Tokenize =====================
def process_and_featurize_split(dc_dataset, tokenizer, max_atoms=64):
    df = dc_dataset.to_dataframe()
    df = df.rename(columns={"X": "smiles"})
    df["smiles"] = df["smiles"].apply(
        lambda x: Chem.MolToSmiles(x) if isinstance(x, Chem.Mol) else x
    )
    df["mol"] = df["smiles"].apply(Chem.MolFromSmiles)
    df = df.dropna(subset=["mol"])
    df["num_atoms"] = df["mol"].apply(lambda m: m.GetNumAtoms())
    df = df[df["num_atoms"] <= max_atoms].dropna().reset_index(drop=True)
    if df.empty:
        return None, None

    target_cols = [col for col in df.columns if col.startswith("y")]
    y_targets_df = df[target_cols]
    mols = df["mol"].tolist()

    print("   - Generating GNN Embeddings...")
    df["tokenized"] = df["smiles"].apply(tokenizer.tokenize)
    dataset_gnn = MaskedMoleculeDataset(
        df["tokenized"].tolist(), mask_ratio_atoms=0, mask_ratio_bonds=0
    )
    loader_gnn = DataLoader(dataset_gnn, batch_size=256, shuffle=False)
    with torch.no_grad():
        gnn_feats = [
            inference_model.get_embedding(batch.to(device)).cpu().numpy()
            for batch in loader_gnn
        ]
    features = {
        "Our GNN": np.concatenate(gnn_feats, axis=0),
        "ECFP": np.array(
            [
                (
                    generate_ecfp(m)
                    if generate_ecfp(m) is not None
                    else np.zeros(2048, dtype=np.int8)
                )
                for m in mols
            ]
        ),
        "MACCS": np.array(
            [
                (
                    generate_maccs(m)
                    if generate_maccs(m) is not None
                    else np.zeros(166, dtype=np.int8)
                )
                for m in mols
            ]
        ),
        "RDKit FP": np.array(
            [
                (
                    generate_rdkit_fp(m)
                    if generate_rdkit_fp(m) is not None
                    else np.zeros(2048, dtype=np.int8)
                )
                for m in mols
            ]
        ),
        "Atom Pairs": np.array(
            [
                (
                    generate_atom_pairs(m)
                    if generate_atom_pairs(m) is not None
                    else np.zeros(2048, dtype=np.int8)
                )
                for m in mols
            ]
        ),
        "2D Descriptors": np.array([generate_rdkit_descriptors(m) for m in mols]),
    }
    return y_targets_df, features


# ===================== Benchmark Loop =====================
BENCHMARK_CONFIG = [
    {"name": "ESOL", "loader": dc.molnet.load_delaney, "task_type": "regression"},
    {"name": "FreeSolv", "loader": dc.molnet.load_freesolv, "task_type": "regression"},
    {"name": "Lipophilicity", "loader": dc.molnet.load_lipo, "task_type": "regression"},
    {"name": "BBBP", "loader": dc.molnet.load_bbbp, "task_type": "classification"},
    {
        "name": "ClinTox",
        "loader": dc.molnet.load_clintox,
        "task_type": "classification",
    },
    {
        "name": "BACE",
        "loader": dc.molnet.load_bace_classification,
        "task_type": "classification",
    },
    {"name": "SIDER", "loader": dc.molnet.load_sider, "task_type": "classification"},
]

all_results = []
for config in BENCHMARK_CONFIG:
    print(f"\n===== Processing Dataset: {config['name']} =====")
    tasks, (train_set, _, test_set), _ = config["loader"](
        featurizer="Raw", splitter="scaffold", reload=True
    )

    y_train_df, X_train_features = process_and_featurize_split(train_set, tokenizer)
    y_test_df, X_test_features = process_and_featurize_split(test_set, tokenizer)

    if y_train_df is None or y_test_df is None:
        print(f"Skipping {config['name']} due to no valid molecules after filtering.")
        continue

    metric_accumulator = {}
    n_tasks = len(y_train_df.columns)

    for task_name in y_train_df.columns:
        y_train = y_train_df[task_name].values
        y_test = y_test_df[task_name].values

        for method_name, X_train in X_train_features.items():
            X_test = X_test_features[method_name]
            metrics = run_evaluation(
                X_train, y_train, X_test, y_test, config["task_type"], method_name
            )
            for k, v in metrics.items():
                metric_accumulator.setdefault(method_name, {}).setdefault(k, []).append(
                    float(v)
                )

    if metric_accumulator:
        result_entry = {
            "dataset": config["name"],
            "task": config["task_type"],
            "metrics": {},
        }
        for method, scores in metric_accumulator.items():
            for metric_name, values in scores.items():
                result_entry["metrics"].setdefault(metric_name, {})[method] = float(
                    np.mean(values)
                )
        all_results.append(result_entry)

with open("results/benchmark_results.json", "w") as f:
    json.dump(all_results, f, indent=2)
print("\nSaved final benchmark results to results/benchmark_results.json")
