import os
import torch
import pandas as pd
import numpy as np
from torch_geometric.loader import DataLoader
from argparse import ArgumentParser
from multiprocessing import Pool, cpu_count, set_start_method
from tqdm import tqdm

from config import *
from dataset import MaskedMoleculeDataset
from lightning_model import GraphMoleculeLightning
from tokenizer import SMILESTokenizer
from utils import has_max_64_atoms

# Global vars for each process
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
    filtered = (
        chunk_df[chunk_df["valid_smiles"]]
        .drop(columns=["valid_smiles"])
        .reset_index(drop=True)
    )

    if len(filtered) == 0:
        return None

    smiles_list = filtered["smiles"].tolist()
    tokenized = [_tokenizer.tokenize(s) for s in smiles_list]
    dataset = MaskedMoleculeDataset(
        tokenized, mask_ratio_atoms=0.0, mask_ratio_bonds=0.0
    )
    loader = DataLoader(dataset, batch_size=64, shuffle=False)

    chunk_embeddings = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(_device)
            embeddings = _model.get_embedding(batch)
            chunk_embeddings.append(embeddings.cpu().numpy())

    final_embeddings = np.concatenate(chunk_embeddings, axis=0)

    if final_embeddings.shape[0] != len(smiles_list):
        print(
            f"[Warning] Mismatch: {final_embeddings.shape[0]} embeddings vs {len(smiles_list)} SMILES"
        )
        return None

    chunk_df = pd.DataFrame(
        {"smiles": smiles_list, "embedding": list(final_embeddings)}
    )

    output_file = f"embeddings/chunk_{chunk_id:03d}.pkl"
    chunk_df.to_pickle(output_file)
    return output_file


def chunk_generator(file_path, chunk_size):
    with pd.read_csv(file_path, chunksize=chunk_size) as reader:
        for i, chunk in enumerate(reader):
            yield (i, chunk)


def main():
    parser = ArgumentParser()
    parser.add_argument("--model-file", type=str, default="model/final_model.ckpt")
    parser.add_argument("--data-file", type=str, default="data/canonical_smiles.csv")
    parser.add_argument("--chunk-size", type=int, default=5000)
    parser.add_argument(
        "--output", type=str, default="embeddings/molecule_embeddings.pkl"
    )
    args = parser.parse_args()

    os.makedirs("embeddings", exist_ok=True)

    print(f"🧠 Using {cpu_count() // 3} processes for multiprocessing...")

    with Pool(
        processes=cpu_count() // 3, initializer=init_worker, initargs=(args.model_file,)
    ) as pool:
        output_files = list(
            tqdm(
                pool.imap(
                    process_chunk, chunk_generator(args.data_file, args.chunk_size)
                ),
                desc="Processing chunks",
            )
        )

    # Merge results
    print("📦 Merging chunk results...")
    all_dfs = [pd.read_pickle(f) for f in output_files if f is not None]
    final_df = pd.concat(all_dfs, ignore_index=True)
    final_df.to_pickle(args.output)

    print(f"✅ Done. Saved final embeddings to: {args.output}")
    print(
        f"📊 Total molecules: {len(final_df)} | Shape: {np.array(final_df.embedding.tolist()).shape}"
    )


if __name__ == "__main__":
    try:
        set_start_method("spawn")
    except RuntimeError:
        pass
    main()
