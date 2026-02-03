# test/generate_embeddings_GATv2.py

import os
import torch
import pandas as pd
import numpy as np
from torch_geometric.loader import DataLoader
from argparse import ArgumentParser
from multiprocessing import Pool, cpu_count, set_start_method
from tqdm import tqdm

from train.lightning_model import GraphMoleculeLightning
from train.data_module import MoleculeDataModule
from train.utils import smiles_to_graph  # same function used in GAT datamodule

_model = None
_device = None


def init_worker(model_file):
    global _model, _device
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lightning_model = GraphMoleculeLightning.load_from_checkpoint(model_file)
    _model = lightning_model.model.to(_device).eval()


def process_chunk(args):
    chunk_id, chunk_df = args
    global _model, _device

    smiles_list = chunk_df["smiles"].tolist()
    dataset = []
    valid_smiles = []

    for smi in smiles_list:
        data = smiles_to_graph(smi)
        if data is not None:
            dataset.append(data)
            valid_smiles.append(smi)

    if len(dataset) == 0:
        return None

    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    chunk_embeddings = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(_device)
            embeddings = _model.get_embedding(batch)
            chunk_embeddings.append(embeddings.cpu().numpy())

    final_embeddings = np.concatenate(chunk_embeddings, axis=0)

    df = pd.DataFrame({
        "smiles": valid_smiles,
        "embedding": list(final_embeddings)
    })

    output_file = f"embeddings/chunk_{chunk_id:03d}.pkl"
    df.to_pickle(output_file)
    return output_file


def chunk_generator(file_path, chunk_size):
    with pd.read_csv(file_path, chunksize=chunk_size) as reader:
        for i, chunk in enumerate(reader):
            yield (i, chunk)


def main():
    parser = ArgumentParser()
    parser.add_argument("--model-file", type=str, default="results/models/final_model.ckpt")
    parser.add_argument("--data-file", type=str, default="data/canonical_smiles.csv")
    parser.add_argument("--chunk-size", type=int, default=5000)
    parser.add_argument("--output", type=str, default="embeddings/gatv2_embeddings.pkl")
    args = parser.parse_args()

    os.makedirs("embeddings", exist_ok=True)

    print(f"🧠 Using {cpu_count() // 3} processes")

    with Pool(
        processes=cpu_count() // 3,
        initializer=init_worker,
        initargs=(args.model_file,)
    ) as pool:
        output_files = list(
            tqdm(
                pool.imap(
                    process_chunk,
                    chunk_generator(args.data_file, args.chunk_size)
                ),
                desc="Processing chunks"
            )
        )

    print("📦 Merging chunks...")
    dfs = [pd.read_pickle(f) for f in output_files if f is not None]
    final_df = pd.concat(dfs, ignore_index=True)
    final_df.to_pickle(args.output)

    print(f"✅ Saved embeddings to: {args.output}")
    print(f"Shape: {np.array(final_df.embedding.tolist()).shape}")


if __name__ == "__main__":
    try:
        set_start_method("spawn")
    except RuntimeError:
        pass
    main()