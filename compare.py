import os
import warnings
from argparse import ArgumentParser

import deepchem as dc
import numpy as np
import pandas as pd
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import (AllChem, Descriptors, MACCSkeys,
                        rdMolDescriptors)
from rdkit.DataStructs import cDataStructs
from sklearn.metrics import (accuracy_score, f1_score, mean_squared_error,
                             precision_score, r2_score, recall_score,
                             roc_auc_score)
from torch_geometric.loader import DataLoader
from xgboost import XGBClassifier, XGBRegressor

# --- Local Project Imports ---
# Ensure these files are in your project directory
from config import *
from dataset import MaskedMoleculeDataset
from model import GraphMoleculeModel
from tokenizer import SMILESTokenizer
from utils import create_graph_data_for_inference

# =============================================================================
# 1. SETUP & INITIALIZATION
# =============================================================================

# Suppress RDKit warnings for cleaner output
RDLogger.DisableLog('rdApp.WARNING')
warnings.filterwarnings("ignore", category=UserWarning)

# Create results directory if it doesn't exist
if not os.path.exists("results"):
    os.makedirs("results")

# --- Command-Line Argument Parsing ---
parser = ArgumentParser(description="Run MoleculeNet benchmark for a pre-trained GNN model.")
parser.add_argument("--model-file", type=str, required=True, help="Path to the final GNN model file (.pt or .pth)")
args = parser.parse_args()

# --- Model and Tokenizer Loading ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
inference_model = GraphMoleculeModel(hidden_dim=hidden_dim, num_layers=num_layers, num_heads=num_heads, ATOM_VOCAB_SIZE=ATOM_VOCAB_SIZE, BOND_VOCAB_SIZE=BOND_VOCAB_SIZE, device=device)
inference_model.load_state_dict(torch.load(args.model_file, map_location=device))
inference_model.eval()  # Set model to evaluation mode
tokenizer = SMILESTokenizer()

print(f"Model weights loaded successfully from {args.model_file} and running on {device}.")
print("=" * 70)
print(" " * 10 + "RUNNING FINAL BENCHMARK ON MOLECULENET SUITE")
print("=" * 70)


# =============================================================================
# 2. FEATURIZER FUNCTIONS
# =============================================================================

def _fp_to_np(fp, n_bits):
    """Helper to convert an RDKit fingerprint to a NumPy array."""
    arr = np.zeros(n_bits, dtype=np.int8)
    cDataStructs.ConvertToNumpyArray(fp, arr)
    return arr

def generate_ecfp(mol, radius=2, n_bits=2048):
    """Generates Morgan Fingerprint (ECFP4)."""
    if mol is None: return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    return _fp_to_np(fp, n_bits)

def generate_maccs(mol):
    """Generates MACCS Keys."""
    if mol is None: return None
    fp = MACCSkeys.GenMACCSKeys(mol)
    return _fp_to_np(fp, 166)  # MACCS keys are a fixed 166-bit length

def generate_rdkit_fp(mol, n_bits=2048):
    """Generates RDKit Path-Based Fingerprint."""
    if mol is None: return None
    fp = Chem.RDKFingerprint(mol, fpSize=n_bits)
    return _fp_to_np(fp, n_bits)

def generate_atom_pairs(mol, n_bits=2048):
    """Generates Atom-Pair Fingerprint."""
    if mol is None: return None
    fp = rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(mol, nBits=n_bits)
    return _fp_to_np(fp, n_bits)

_desc_list = [d[0] for d in Descriptors._descList]
_num_descriptors = len(_desc_list)
def generate_rdkit_descriptors(mol):
    """
    Generates RDKit 2D descriptors, robustly handling calculation errors.
    This function sanitizes the output to ensure it's always a float array.
    """
    if mol is None:
        return np.full(_num_descriptors, np.nan, dtype=np.float64)
    try:
        descs = Descriptors.CalcMolDescriptors(mol)
        # Sanitize: Replace any non-numeric or invalid float values with np.nan
        sanitized_descs = [d if isinstance(d, (int, float)) and np.isfinite(d) else np.nan for d in descs]
        return np.array(sanitized_descs, dtype=np.float64)
    except:
        # Catch any other major error and return a NaN vector
        return np.full(_num_descriptors, np.nan, dtype=np.float64)


# =============================================================================
# 3. EVALUATION LOGIC
# =============================================================================

def run_evaluation(X_train, y_train, X_test, y_test, task_type, method_name):
    """Trains an XGBoost model and evaluates its performance."""
    # NaN value handling is only necessary for 2D Descriptors
    if method_name == "2D Descriptors":
        train_nan_mask = ~np.isnan(X_train).all(axis=1)
        X_train, y_train = X_train[train_nan_mask], y_train[train_nan_mask]
        test_nan_mask = ~np.isnan(X_test).all(axis=1)
        X_test, y_test = X_test[test_nan_mask], y_test[test_nan_mask]

    if len(X_train) == 0 or len(X_test) == 0:
        return {} # Not enough data to train or evaluate

    if task_type == "regression":
        model = XGBRegressor(random_state=42, n_jobs=-1, objective='reg:squarederror')
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        return {"R2": r2_score(y_test, y_pred), "RMSE": np.sqrt(mean_squared_error(y_test, y_pred))}
    else:  # classification
        # Handle class imbalance, common in tox/bio datasets
        scale_pos_weight = (np.sum(y_train == 0) / np.sum(y_train == 1)) if np.sum(y_train == 1) > 0 else 1
        model = XGBClassifier(random_state=42, n_jobs=-1, use_label_encoder=False, eval_metric='logloss', scale_pos_weight=scale_pos_weight)
        model.fit(X_train, y_train)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba > 0.5).astype(int)
        return {
            "AUC": roc_auc_score(y_test, y_pred_proba), "Accuracy": accuracy_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred, zero_division=0), "Precision": precision_score(y_test, y_pred, zero_division=0),
            "F1": f1_score(y_test, y_pred, zero_division=0),
        }


# =============================================================================
# 4. DATA PROCESSING
# =============================================================================

def process_and_featurize_split(dc_dataset, tokenizer, max_atoms=64):
    """Cleans a dataset split and generates all required feature sets."""
    df = dc_dataset.to_dataframe()
    target_cols = [col for col in df.columns if col.startswith("y")]
    df = df.rename(columns={"X": "smiles"})
    df["smiles"] = df["smiles"].apply(lambda x: Chem.MolToSmiles(x) if isinstance(x, Chem.Mol) else x)
    
    # Filter out invalid SMILES, molecules over max_atoms, and tasks with NaN labels
    df["mol"] = df["smiles"].apply(Chem.MolFromSmiles)
    df = df.dropna(subset=['mol'])
    df["num_atoms"] = df["mol"].apply(lambda m: m.GetNumAtoms())
    df = df[df["num_atoms"] <= max_atoms]
    df = df.dropna(subset=target_cols).reset_index(drop=True)

    if df.empty:
        return None, None

    y_targets_df = df[target_cols]
    mols = df["mol"].tolist()
    features = {}

    # --- Feature Generation ---
    # 1. GNN Embeddings
    print("   - Generating GNN Embeddings...")
    df["tokenized"] = df["smiles"].apply(tokenizer.tokenize)
    dataset_gnn = MaskedMoleculeDataset(df["tokenized"].tolist())
    loader_gnn = DataLoader(dataset_gnn, batch_size=256, shuffle=False)
    with torch.no_grad():
        features["Our GNN"] = np.concatenate([inference_model.get_embedding(batch.to(device)).cpu().numpy() for batch in loader_gnn], axis=0)

    # 2. Traditional Fingerprints & Descriptors
    print("   - Generating traditional fingerprints & descriptors...")
    # This robustly handles cases where a featurizer returns None by substituting a zero-vector,
    # which prevents the creation of a NumPy array with dtype=object.
    n_bits = 2048
    features["ECFP"] = np.array([fp if (fp := generate_ecfp(m, n_bits=n_bits)) is not None else np.zeros(n_bits, dtype=np.int8) for m in mols])
    features["MACCS"] = np.array([fp if (fp := generate_maccs(m)) is not None else np.zeros(166, dtype=np.int8) for m in mols])
    features["RDKit FP"] = np.array([fp if (fp := generate_rdkit_fp(m, n_bits=n_bits)) is not None else np.zeros(n_bits, dtype=np.int8) for m in mols])
    features["Atom Pairs"] = np.array([fp if (fp := generate_atom_pairs(m, n_bits=n_bits)) is not None else np.zeros(n_bits, dtype=np.int8) for m in mols])
    features["2D Descriptors"] = np.array([generate_rdkit_descriptors(m) for m in mols])
    
    return y_targets_df, features


# =============================================================================
# 5. MAIN EXECUTION LOOP
# =============================================================================

BENCHMARK_CONFIG = [
    {"name": "ESOL", "loader": dc.molnet.load_delaney, "task_type": "regression"},
    {"name": "FreeSolv", "loader": dc.molnet.load_freesolv, "task_type": "regression"},
    {"name": "Lipophilicity", "loader": dc.molnet.load_lipo, "task_type": "regression"},
    {"name": "BBBP", "loader": dc.molnet.load_bbbp, "task_type": "classification"},
    {"name": "ClinTox", "loader": dc.molnet.load_clintox, "task_type": "classification"},
    {"name": "BACE", "loader": dc.molnet.load_bace_classification, "task_type": "classification"},
    {"name": "SIDER", "loader": dc.molnet.load_sider, "task_type": "classification"},
]

all_results = []
for config in BENCHMARK_CONFIG:
    print(f"\n===== Processing Dataset: {config['name']} =====")
    tasks, (train_set, _, test_set), _ = config["loader"](featurizer="Raw", splitter="scaffold")

    print("Processing & Featurizing Train Set...")
    y_train_df, X_train_features = process_and_featurize_split(train_set, tokenizer)
    print("Processing & Featurizing Test Set...")
    y_test_df, X_test_features = process_and_featurize_split(test_set, tokenizer)

    if y_train_df is None or y_test_df is None:
        print(f"Skipping {config['name']} due to no valid molecules after filtering.")
        continue

    # Evaluate each task (for multi-task datasets)
    for task_name in y_train_df.columns:
        print(f"--- Evaluating Task: {task_name} ---")
        y_train = y_train_df[task_name].values
        y_test = y_test_df[task_name].values

        # Evaluate each featurization method
        for method_name, X_train in X_train_features.items():
            print(f"    - Method: {method_name}")
            X_test = X_test_features[method_name]
            metrics = run_evaluation(X_train, y_train, X_test, y_test, config["task_type"], method_name)
            if metrics:
                # Store raw results, including dataset and task name
                all_results.append({"Dataset": f"{config['name']}-{task_name}", "Method": method_name, **metrics})

# =============================================================================
# 6. RESULTS AGGREGATION & DISPLAY
# =============================================================================

print(f"\nCompleted evaluation. Aggregating results...")
results_df = pd.DataFrame(all_results)
results_df['DatasetName'] = results_df['Dataset'].apply(lambda x: x.split('-')[0].strip())

metric_cols = ["R2", "RMSE", "AUC", "Accuracy", "F1"]
method_order = ["Our GNN", "ECFP", "RDKit FP", "Atom Pairs", "MACCS", "2D Descriptors"]

# Aggregate results by taking the mean across all tasks for a given dataset (e.g., SIDER)
aggregated_results = results_df.groupby(['DatasetName', 'Method'])[metric_cols].mean()

# Pivot the table to have methods as columns for easy comparison
final_table = aggregated_results.unstack(level='Method')

# Reorder method columns to our desired order
available_methods = [m for m in method_order if m in final_table.columns.get_level_values('Method')]
final_table = final_table.reindex(columns=available_methods, level='Method')

# Drop metric columns that are entirely empty (e.g., R2 for classification tasks)
final_table = final_table.dropna(axis=1, how='all')

# Sort datasets alphabetically for consistent output
final_table.sort_index(inplace=True)

# --- Display and Save Final Table ---
print("\n\n" + "=" * 120)
print(" " * 35 + "FINAL AGGREGATED BENCHMARK RESULTS (Scaffold Split)")
print(" " * 45 + "(Mean score across all tasks)")
print("=" * 120)
print(final_table.to_string(float_format="%.4f", na_rep="-"))
print("=" * 120)

output_path = "results/benchmark_results.csv"
final_table.to_csv(output_path, float_format="%.4f")
print(f"\nAggregated results saved to {output_path}")