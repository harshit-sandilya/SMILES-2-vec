from argparse import ArgumentParser

import torch
from sklearn.metrics.pairwise import cosine_similarity

from model import GraphMoleculeModel
from tokenizer import SMILESTokenizer
from utils import get_single_embedding
from config import *

parser = ArgumentParser()
parser.add_argument(
    "--model-file", type=str, required=True, help="Path to the final model file"
)
args = parser.parse_args()
model_file = args.model_file

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
inference_model = GraphMoleculeModel(hidden_dim=hidden_dim,num_layers=num_layers,num_heads=num_heads,ATOM_VOCAB_SIZE=ATOM_VOCAB_SIZE,BOND_VOCAB_SIZE=BOND_VOCAB_SIZE,device=device)
inference_model.load_state_dict(torch.load(model_file, map_location=device))
inference_model.eval()
print("Model weights loaded successfully.")
tokenizer = SMILESTokenizer()

# Pair 1: Very Similar (Homologous Series: adding one carbon)
mol_toluene = "Cc1ccccc1"
mol_ethylbenzene = "CCc1ccccc1"

# Pair 2: Isomers (Same formula C3H8O, different connectivity)
mol_propanol = "CCCO"
mol_isopropanol = "CC(C)O"

# Pair 3: Very Dissimilar (Different size, function, scaffold)
mol_aspirin = "CC(=O)OC1=CC=CC=C1C(=O)O"
mol_glucose = "C(C1C(C(C(C(O1)O)O)O)O)O"

# Pair 4: Stereoisomers (Identical graph, different 3D shape)
mol_r_alanine = "C[C@H](N)C(=O)O"
mol_s_alanine = "C[C@@H](N)C(=O)O"

# Pair 5: Bioisosteres (Similar function/scaffold, different electronics)
mol_benzoic_acid = "c1ccc(C(=O)O)cc1"
mol_nicotinic_acid = "c1cnc(C(=O)O)cc1"

# Pair 6: Functional Group Isomers (Same formula, different function)
mol_ethanol = "CCO"
mol_dimethyl_ether = "COC"

emb_toluene = get_single_embedding(mol_toluene, inference_model, tokenizer, device)
emb_ethylbenzene = get_single_embedding(
    mol_ethylbenzene, inference_model, tokenizer, device
)
emb_propanol = get_single_embedding(mol_propanol, inference_model, tokenizer, device)
emb_isopropanol = get_single_embedding(
    mol_isopropanol, inference_model, tokenizer, device
)
emb_aspirin = get_single_embedding(mol_aspirin, inference_model, tokenizer, device)
emb_glucose = get_single_embedding(mol_glucose, inference_model, tokenizer, device)
emb_r_alanine = get_single_embedding(mol_r_alanine, inference_model, tokenizer, device)
emb_s_alanine = get_single_embedding(mol_s_alanine, inference_model, tokenizer, device)
emb_benzoic_acid = get_single_embedding(
    mol_benzoic_acid, inference_model, tokenizer, device
)
emb_nicotinic_acid = get_single_embedding(
    mol_nicotinic_acid, inference_model, tokenizer, device
)
emb_ethanol = get_single_embedding(mol_ethanol, inference_model, tokenizer, device)
emb_dimethyl_ether = get_single_embedding(
    mol_dimethyl_ether, inference_model, tokenizer, device
)

sim_homologous = cosine_similarity(emb_toluene, emb_ethylbenzene)[0, 0]
sim_isomers = cosine_similarity(emb_propanol, emb_isopropanol)[0, 0]
sim_dissimilar = cosine_similarity(emb_aspirin, emb_glucose)[0, 0]
sim_stereoisomers = cosine_similarity(emb_r_alanine, emb_s_alanine)[0, 0]
sim_bioisosteres = cosine_similarity(emb_benzoic_acid, emb_nicotinic_acid)[0, 0]
sim_functional_isomers = cosine_similarity(emb_ethanol, emb_dimethyl_ether)[0, 0]

print("\n--- Cosine Similarity Results ---")
print(f"Toluene vs. Ethylbenzene (Homologous) (> 0.95): {sim_homologous:.4f}")
print(f"Propanol vs. Isopropanol (Isomers) (0.90 - 0.98):   {sim_isomers:.4f}")
print(f"Aspirin vs. Glucose (Dissimilar) (< 0.40):    {sim_dissimilar:.4f}")
print(f"R-Alanine vs. S-Alanine (Stereoisomers) (> 0.95): {sim_stereoisomers:.4f}")
print(
    f"Benzoic Acid vs. Nicotinic Acid (Bioisosteres) (0.75 - 0.90): {sim_bioisosteres:.4f}"
)
print(
    f"Ethanol vs. Dimethyl Ether (Functional Isomers) (0.40 - 0.65): {sim_functional_isomers:.4f}"
)
print("-----------------------------------")
