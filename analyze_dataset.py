import argparse
import os
from functools import partial
from multiprocessing import Pool, cpu_count

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import Descriptors
from tqdm import tqdm


def process_chunk(chunk, functional_groups):
    chunk_aggregated_data = {
        "molecular_weight": [],
        "logp": [],
        "hbd": [],
        "hba": [],
        "num_atoms": [],
        "num_heavy_atoms": [],
        "num_rings": [],
        "tpsa": [],
    }
    chunk_fg_counts = {name: 0 for name in functional_groups.keys()}
    chunk["rdkit_mol"] = [Chem.MolFromSmiles(s) for s in chunk["smiles"]]
    chunk.dropna(subset=["rdkit_mol"], inplace=True)
    if chunk.empty:
        return None
    chunk_aggregated_data["molecular_weight"].extend(chunk["molecular_weight"])
    chunk_aggregated_data["logp"].extend(chunk["logp"])
    chunk_aggregated_data["hbd"].extend(chunk["hbd"])
    chunk_aggregated_data["hba"].extend(chunk["hba"])
    mols = chunk["rdkit_mol"]
    chunk_aggregated_data["num_atoms"].extend([m.GetNumAtoms() for m in mols])
    chunk_aggregated_data["num_heavy_atoms"].extend(
        [m.GetNumHeavyAtoms() for m in mols]
    )
    chunk_aggregated_data["num_rings"].extend(
        [m.GetRingInfo().NumRings() for m in mols]
    )
    chunk_aggregated_data["tpsa"].extend([Descriptors.TPSA(m) for m in mols])
    fg_patterns = {
        name: Chem.MolFromSmarts(smarts) for name, smarts in functional_groups.items()
    }
    for mol in mols:
        for name, pattern in fg_patterns.items():
            if mol.HasSubstructMatch(pattern):
                chunk_fg_counts[name] += 1
    return chunk_aggregated_data, chunk_fg_counts, len(chunk)


class DatasetAnalyzer:
    def __init__(self, data_path, plots_dir, chunksize=10000, num_workers=None):
        self.data_path = data_path
        self.plots_dir = plots_dir
        self.chunksize = chunksize
        self.num_workers = num_workers if num_workers else cpu_count()
        self.aggregated_data = {
            "molecular_weight": [],
            "logp": [],
            "hbd": [],
            "hba": [],
            "num_atoms": [],
            "num_heavy_atoms": [],
            "num_rings": [],
            "tpsa": [],
        }
        self.functional_groups = {
            "Hydroxyl": "[#6][OH]",
            "Carbonyl": "[#6]C=O",
            "Carboxylic Acid": "C(=O)[O;H1]",
            "Amine": "[NX3;H2;!$(NC=O)]",
            "Amide": "C(=O)N",
            "Ester": "C(=O)OC",
            "Ether": "[OD2]([#6])[#6]",
            "Sulfonamide": "S(=O)(=O)N",
            "Pyridine": "c1ncccc1",
            "Benzene": "c1ccccc1",
            "Halogen (F,Cl,Br,I)": "[F,Cl,Br,I]",
        }
        self.fg_counts = {name: 0 for name in self.functional_groups.keys()}
        if not os.path.exists(self.plots_dir):
            os.makedirs(self.plots_dir)
            print(f"Created directory: {self.plots_dir}")

    def run_full_analysis(self):
        print(f"Starting analysis of {self.data_path} with {self.num_workers} workers.")
        print(f"Processing in chunks of {self.chunksize} rows.")
        total_rows = 0
        valid_molecules = 0
        try:
            num_lines = sum(1 for _ in open(self.data_path)) - 1
            total_rows = num_lines
            num_chunks = (num_lines // self.chunksize) + 1
            if num_lines == 0:
                print("Error: The data file is empty.")
                return
            reader = pd.read_csv(self.data_path, chunksize=self.chunksize)
            worker_func = partial(
                process_chunk, functional_groups=self.functional_groups
            )
            with Pool(processes=self.num_workers) as pool:
                results_iterator = pool.imap(worker_func, reader)
                pbar = tqdm(
                    results_iterator, total=num_chunks, desc="Processing chunks"
                )
                for result in pbar:
                    if result is None:
                        continue
                    chunk_data, chunk_fg_counts, valid_count = result
                    valid_molecules += valid_count
                    for key, value_list in chunk_data.items():
                        self.aggregated_data[key].extend(value_list)
                    for name, count in chunk_fg_counts.items():
                        self.fg_counts[name] += count
        except FileNotFoundError:
            print(f"Error: The file {self.data_path} was not found.")
            return
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return
        print("\n" + "-" * 30)
        print("File processing complete.")
        print(f"Successfully processed {valid_molecules} of {total_rows} total rows.")
        print("-" * 30)
        if valid_molecules > 0:
            agg_df = pd.DataFrame(self.aggregated_data)
            self.plot_property_distributions(agg_df)
            self.plot_correlation_heatmap(agg_df)
            self.plot_functional_group_analysis()
            print(f"Analysis complete. Plots saved to '{self.plots_dir}'.")
        else:
            print("No valid molecules were found to analyze.")

    def plot_property_distributions(self, df):
        print("Plotting property distributions...")
        for prop in df.columns:
            plt.figure(figsize=(10, 6))
            sns.histplot(df[prop], kde=True, bins=50)
            plt.title(f'Distribution of {prop.replace("_", " ").title()}', fontsize=16)
            plt.xlabel(prop.title(), fontsize=12)
            plt.ylabel("Frequency", fontsize=12)
            plt.grid(axis="y", linestyle="--", alpha=0.7)
            plot_path = os.path.join(self.plots_dir, f"{prop}_distribution.png")
            plt.savefig(plot_path)
            plt.close()
            print(f"  - Saved plot: {plot_path}")
        print("-" * 30)

    def plot_functional_group_analysis(self):
        print("Analyzing functional groups...")
        fg_df = pd.DataFrame(
            list(self.fg_counts.items()), columns=["Functional Group", "Count"]
        )
        fg_df = fg_df.sort_values("Count", ascending=False).reset_index(drop=True)
        plt.figure(figsize=(12, 8))
        sns.barplot(x="Count", y="Functional Group", data=fg_df.head(15))
        plt.title("Frequency of Common Functional Groups", fontsize=16)
        plt.xlabel("Number of Molecules with Group", fontsize=12)
        plt.ylabel("Functional Group", fontsize=12)
        plt.grid(axis="x", linestyle="--", alpha=0.7)
        plot_path = os.path.join(self.plots_dir, "functional_group_analysis.png")
        plt.savefig(plot_path, bbox_inches="tight")
        plt.close()
        print(f"  - Saved plot: {plot_path}")
        print("Functional group counts:")
        print(fg_df)
        print("-" * 30)

    def plot_correlation_heatmap(self, df):
        print("Generating property correlation heatmap...")
        correlation_matrix = df.corr()
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5
        )
        plt.title("Correlation Matrix of Molecular Properties", fontsize=16)
        plot_path = os.path.join(self.plots_dir, "property_correlation_heatmap.png")
        plt.savefig(plot_path, bbox_inches="tight")
        plt.close()
        print(f"  - Saved plot: {plot_path}")
        print("-" * 30)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Perform memory-efficient, parallel analysis on large cheminformatics datasets.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--data-file",
        default="data/smiles.csv",
        type=str,
        help="Path to the input CSV data file.",
    )
    parser.add_argument(
        "--plots-dir",
        default="plots",
        type=str,
        help="Directory to save analysis plots.",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=10000,
        help="Number of rows to process at a time (default: 10000).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of worker processes to use (default: all CPU cores).",
    )
    args = parser.parse_args()
    analyzer = DatasetAnalyzer(
        data_path=args.data_file,
        plots_dir=args.plots_dir,
        chunksize=args.chunksize,
        num_workers=args.num_workers,
    )
    analyzer.run_full_analysis()
