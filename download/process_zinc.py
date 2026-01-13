import argparse
import csv
import os

from tqdm import tqdm

BASE_DATA_DIR = "data"

def main(input_folder, output_path):
    # Ensure output is written under data/
    output_csv = (
        output_path
        if os.path.isabs(output_path)
        else os.path.join(BASE_DATA_DIR, output_path)
    )


    output_dir = os.path.dirname(output_csv)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Created directory: {output_dir}")

    if os.path.exists(output_csv):
        os.remove(output_csv)
        print(f"Removed existing file: {output_csv}")

    smi_files = [f for f in os.listdir(input_folder) if f.endswith(".smi")]

    with open(output_csv, mode="w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["zinc_id", "smiles"])

        for smi_file in tqdm(smi_files, desc="Processing files", unit="file"):
            file_path = os.path.join(input_folder, smi_file)
            try:
                with open(file_path, "r") as infile:
                    for line in infile:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            smiles, zinc_id = line.split()
                            if zinc_id == "zinc_id":
                                continue
                            writer.writerow([zinc_id, smiles])
                        except ValueError:
                            print(f"Skipping malformed line in {smi_file}: {line}")
            except Exception as e:
                print(f"Error reading {smi_file}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process .smi files and write to CSV.")
    parser.add_argument(
        "--input-folder",
        default="ZINC",
        help="Path to the input folder containing .smi files",
    )
    parser.add_argument(
    "--output-path",
    default="zinc_smiles.csv",
    help="Output CSV filename (saved under data/ by default)",
    )

    args = parser.parse_args()
    main(args.input_folder, args.output_path)
