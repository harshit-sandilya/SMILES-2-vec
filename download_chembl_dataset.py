import argparse
import os

import pandas as pd
from chembl_webresource_client.new_client import new_client
from tqdm import tqdm


def save_batch_to_csv(data, filename, write_header):
    if not data:
        return
    df_batch = pd.DataFrame(data)
    df_batch.to_csv(filename, mode="a", index=False, header=write_header)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download molecules from ChEMBL in a memory-efficient way.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "-m",
        "--molecules",
        type=int,
        default=None,
        help="Total number of molecules to download (default: all available).",
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        default=5000,
        help="Number of molecules to process before saving to disk (default: 5000).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="data/smiles.csv",
        help="Path for the output CSV file (default: data/chembl_molecules.csv).",
    )
    args = parser.parse_args()

    TOTAL_MOLECULES_TO_DOWNLOAD = args.molecules
    BATCH_SIZE = args.batch_size
    OUTPUT_FILENAME = args.output

    output_dir = os.path.dirname(OUTPUT_FILENAME)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    if os.path.exists(OUTPUT_FILENAME):
        os.remove(OUTPUT_FILENAME)
        print(f"Removed existing file: {OUTPUT_FILENAME}")

    print("Connecting to ChEMBL...")
    molecule_api = new_client.molecule
    print("Connection successful.")

    query_limit_text = (
        "all"
        if not TOTAL_MOLECULES_TO_DOWNLOAD
        else f"up to {TOTAL_MOLECULES_TO_DOWNLOAD}"
    )
    print(f"Querying for {query_limit_text} molecules...")

    molecules_query = molecule_api.all().only(
        "molecule_chembl_id", "molecule_structures", "molecule_properties"
    )

    if TOTAL_MOLECULES_TO_DOWNLOAD:
        molecules_query = molecules_query[:TOTAL_MOLECULES_TO_DOWNLOAD]

    batch_data = []
    total_processed = 0
    header_written = False

    print(f"Downloading and processing data in batches of {BATCH_SIZE}...")
    for mol in tqdm(molecules_query, desc="Processing molecules"):
        if not mol.get("molecule_structures") or not mol.get("molecule_properties"):
            continue
        chembl_id = mol["molecule_chembl_id"]
        smiles = mol["molecule_structures"]["canonical_smiles"]
        props = mol["molecule_properties"]
        mol_weight = props.get("mw_freebase")
        logp = props.get("alogp")
        hbd = props.get("hbd")
        hba = props.get("hba")
        batch_data.append(
            {
                "chembl_id": chembl_id,
                "smiles": smiles,
                "molecular_weight": mol_weight,
                "logp": logp,
                "hbd": hbd,
                "hba": hba,
            }
        )
        if len(batch_data) >= BATCH_SIZE:
            save_batch_to_csv(batch_data, OUTPUT_FILENAME, not header_written)
            header_written = True
            total_processed += len(batch_data)
            batch_data = []

    if batch_data:
        print(f"Saving final batch of {len(batch_data)} molecules...")
        save_batch_to_csv(batch_data, OUTPUT_FILENAME, not header_written)
        total_processed += len(batch_data)

    print("-" * 30)
    if total_processed > 0:
        print(
            f"Download complete. Successfully processed and saved {total_processed} molecules."
        )
        print(f"Your data is ready in: {OUTPUT_FILENAME}")
    else:
        print("No data was downloaded. Please check your query or connection.")
    print("-" * 30)
