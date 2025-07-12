import argparse
import os

import pandas as pd
from chembl_webresource_client.new_client import new_client
from tqdm import tqdm

if not os.path.exists("data"):
    os.makedirs("data")

parser = argparse.ArgumentParser(description="Download molecules from ChEMBL.")
parser.add_argument(
    "-m",
    "--molecules",
    type=int,
    default=None,
    help="Number of molecules to download (default: all)",
)
args = parser.parse_args()

NUMBER_OF_MOLECULES_TO_DOWNLOAD = args.molecules
OUTPUT_FILENAME = "data/chembl_data.csv"

print("Connecting to ChEMBL...")
molecule_api = new_client.molecule
print("Connection successful.")

print(f"Querying for {NUMBER_OF_MOLECULES_TO_DOWNLOAD} molecules...")
molecules_query = molecule_api.all().only(
    "molecule_chembl_id", "molecule_structures", "molecule_properties"
)

if NUMBER_OF_MOLECULES_TO_DOWNLOAD:
    molecules_query = molecules_query[:NUMBER_OF_MOLECULES_TO_DOWNLOAD]

processed_data = []

print("Downloading and processing data...")
for mol in tqdm(molecules_query):
    if not mol.get("molecule_structures") or not mol.get("molecule_properties"):
        continue

    chembl_id = mol["molecule_chembl_id"]
    smiles = mol["molecule_structures"]["canonical_smiles"]
    props = mol["molecule_properties"]
    mol_weight = props.get("mw_freebase")
    logp = props.get("alogp")
    hbd = props.get("hbd")
    hba = props.get("hba")

    processed_data.append(
        {
            "chembl_id": chembl_id,
            "smiles": smiles,
            "molecular_weight": mol_weight,
            "logp": logp,
            "hbd": hbd,
            "hba": hba,
        }
    )

print(f"Successfully processed {len(processed_data)} molecules.")

if processed_data:
    print("Converting to DataFrame...")
    df = pd.DataFrame(processed_data)

    print(f"Saving data to {OUTPUT_FILENAME}...")
    df.to_csv(OUTPUT_FILENAME, index=False)

    print("\nDone! Your file is ready.")
    print("First 5 rows of your data:\n")
    print(df.head())
else:
    print("No data was downloaded. Please check your query.")
