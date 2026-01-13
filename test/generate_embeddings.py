import os
import torch
import pandas as pd
import numpy as np
from torch_geometric.loader import DataLoader
from argparse import ArgumentParser
from multiprocessing import Pool, cpu_count, set_start_method
from tqdm import tqdm
from pathlib import Path

from config import *
from download.dataset import MaskedMoleculeDataset
from train.lightning_model import GraphMoleculeLightning
from preprocess.tokenizer import SMILESTokenizer
from train.utils import has_max_64_atoms

# ==============================
# Resolve project paths
# ==============================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
EMBEDDINGS_DIR = RESULTS_DIR / "embeddings"
MODELS_DIR = RESULTS_DIR / "models"

EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

# ==============================
# Global vars for multiprocessing
# ==============================

_model = None
_tokenizer = None
_device = None


def init_worker(model_file):
    global _model, _tokenizer, _device
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    lightning_model = GraphMoleculeLightning.load_from_checkpoint(model_file)
    _model = lightning_model.model.to(_device).eval()
    _tokenizer = SMILESTokenizer()


def process_chunk(args):
    chunk_id, chunk_df = args
    global _model, _tokenizer, _device

    chunk_df["valid_smiles"] = chunk_df["smiles"].apply(has_max_64_atoms)
    filtered = chunk_df[chunk_df["valid_smiles"]].drop(columns=["valid_smiles"])

    if len(filtered) == 0:
        return None

    smiles_list = filtered["smiles"].tolist()
    tokenized = [_tokenizer.tokenize(s) for s in smiles_list]

    dataset = MaskedMoleculeDataset(
        tokenized,
        mask_ratio_atoms=0.0,
        mask_ratio_bonds=0.0,
    )

    loader = DataLoader(dataset, batch_size=64, shuffle=False)

    chunk_embeddings = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(_device)
            emb = _model.get_embedding(batch)
            chunk_embeddings.append(emb.cpu().numpy())

    final_embeddings = np.concatenate(chunk_embeddings, axis=0)

    chunk_df = pd.DataFrame(
        {"smiles": smiles_list, "embedding": list(final_embeddings)}
    )

    output_file = EMBEDDINGS_DIR / f"chunk_{chunk_id:03d}.pkl"
    chunk_df.to_pickle(output_file)

    return output_file


def chunk_generator(file_path, chunk_size):
    for i, chunk in enumerate(pd.read_csv(file_path, chunksize=chunk_size)):
        yield i, chunk


def main():
    parser = ArgumentParser()

    parser.add_argument(
        "--model-file",
        type=str,
        default="final_model.ckpt",
        help="Checkpoint filename (loaded from results/models/)",
    )
    parser.add_argument(
        "--data-file",
        type=str,
        default="canonical_smiles_subset_10k.csv",
        help="CSV filename inside data/",
    )
    parser.add_argument("--chunk-size", type=int, default=5000)
    parser.add_argument(
        "--output",
        type=str,
        default="molecule_embeddings.pkl",
        help="Final output filename inside results/embeddings/",
    )

    args = parser.parse_args()

    model_file = (
        Path(args.model_file)
        if Path(args.model_file).is_absolute()
        else MODELS_DIR / args.model_file
    )

    data_file = (
        Path(args.data_file)
        if Path(args.data_file).is_absolute()
        else DATA_DIR / args.data_file
    )

    print(f"📄 Dataset : {data_file}")
    print(f"🧠 Model   : {model_file}")
    print(f"⚙️  Workers : {cpu_count() // 3}")

    with Pool(
        processes=max(1, cpu_count() // 3),
        initializer=init_worker,
        initargs=(str(model_file),),
    ) as pool:
        output_files = list(
            tqdm(
                pool.imap(
                    process_chunk,
                    chunk_generator(data_file, args.chunk_size),
                ),
                desc="Generating embeddings",
            )
        )

    # Merge chunks
    print("📦 Merging chunk embeddings...")
    all_dfs = [pd.read_pickle(f) for f in output_files if f is not None]
    final_df = pd.concat(all_dfs, ignore_index=True)

    final_output = (
        Path(args.output)
        if Path(args.output).is_absolute()
        else EMBEDDINGS_DIR / args.output
    )

    final_df.to_pickle(final_output)

    print(f"✅ Saved embeddings to: {final_output}")
    print(
        f"📊 Total molecules: {len(final_df)} | "
        f"Embedding shape: {np.stack(final_df.embedding).shape}"
    )


if __name__ == "__main__":
    try:
        set_start_method("spawn")
    except RuntimeError:
        pass
    main()
