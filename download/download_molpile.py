#!/usr/bin/env python3
"""
MolPILE Downloader — Download SMILES from HuggingFace MolPILE dataset
and save as parquet files in data/raw directory.

This script requires MPI and must be run with mpirun/mpiexec.
Source: scikit-fingerprints/MolPILE dataset from HuggingFace Hub

# Run with default
mpirun -np 4 python download_molpile.py

# Override existing data
mpirun -np 4 python download_molpile.py --override

# Custom output directory
mpirun -np 4 python download_molpile.py --out-dir ./custom/path

# Custom batch size (molecules per file)
mpirun -np 4 python download_molpile.py --batch-size 5000
"""

import argparse
import logging
import os
import shutil
from collections import deque
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from mpi4py import MPI

# MPI setup
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
MPI_ANY_SOURCE = MPI.ANY_SOURCE
MPI_ANY_TAG = MPI.ANY_TAG

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format=f"[Rank {rank}] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Message tags for MPI communication
TAG_READY = 1
TAG_TASK = 2
TAG_DONE = 3
TAG_ERROR = 4

TEMP_DIR = "./datasets"


def get_molecule_batch(
    parquet_path: str, batch_size: int, batch_index: int
) -> Iterator[Tuple[str, str]]:
    """
    Get a batch of molecules from the parquet file.

    Args:
        parquet_path: Path to the parquet file
        batch_size: Number of molecules per batch
        batch_index: Index of the batch to retrieve

    Yields:
        Tuple of (id, SMILES) for each molecule in the batch

    Raises:
        Exception: On file read errors
    """
    try:
        pf = pq.ParquetFile(parquet_path)
        total_rows = pf.metadata.num_rows

        start_idx = batch_index * batch_size
        if start_idx >= total_rows:
            return

        end_idx = min(start_idx + batch_size, total_rows)

        rows_seen = 0
        for rg_idx in range(pf.metadata.num_row_groups):
            rg_rows = pf.metadata.row_group(rg_idx).num_rows
            rg_start = rows_seen
            rg_end = rows_seen + rg_rows
            rows_seen = rg_end

            # Skip row groups entirely outside our window
            if rg_end <= start_idx:
                continue
            if rg_start >= end_idx:
                break

            table = pf.read_row_group(rg_idx, columns=["id", "SMILES"])

            # Trim to the exact slice window within this row group
            local_start = max(0, start_idx - rg_start)
            local_end = min(rg_rows, end_idx - rg_start)
            table = table.slice(local_start, local_end - local_start)

            ids = table.column("id").to_pylist()
            smiles_list = table.column("SMILES").to_pylist()

            for mol_id, smiles in zip(ids, smiles_list):
                mol_id = str(mol_id)
                smiles = str(smiles)
                if mol_id and smiles:
                    yield mol_id, smiles

    except Exception as e:
        logger.error(f"Error reading batch {batch_index} from {parquet_path}: {e}")
        raise


def write_parquet(records: List[Dict[str, str]], path: str) -> None:
    """
    Write records to parquet file with schema {"id": str, "SMILES": str}.

    Args:
        records: List of dictionaries with "id" and "SMILES" keys
        path: Output file path
    """
    schema = pa.schema([("id", pa.string()), ("SMILES", pa.string())])
    table = pa.Table.from_pylist(records, schema=schema)
    pq.write_table(table, path)


def download_molpile_parquet() -> str:
    """
    Download MolPILE parquet file from HuggingFace Hub.

    Returns:
        Path to the downloaded parquet file
    """
    filename = "molpile_smiles.parquet"

    logger.info("Downloading MolPILE dataset")

    try:
        parquet_path = hf_hub_download(
            repo_id="scikit-fingerprints/MolPILE",
            repo_type="dataset",
            filename=filename,
            local_dir=TEMP_DIR,
        )
        logger.info(f"Downloaded to: {parquet_path}")
        return parquet_path
    except Exception as e:
        logger.error(f"Failed to download MolPILE dataset: {e}")
        raise


def get_total_molecules(parquet_path: str) -> int:
    """
    Get the total number of molecules in the parquet file.

    Args:
        parquet_path: Path to the parquet file

    Returns:
        Total number of molecules
    """
    pf = pq.ParquetFile(parquet_path)
    return pf.metadata.num_rows


def setup_output_directory(out_dir: str, override: bool) -> int:
    """
    Set up output directory and determine starting point.

    Args:
        out_dir: Output directory path
        override: Whether to override existing files

    Returns:
        start_k: starting index of parquet part
    """
    os.makedirs(out_dir, exist_ok=True)

    if override:
        for f in Path(out_dir).glob("part_*.parquet"):
            f.unlink()
        return 0
    else:
        existing_files = sorted(Path(out_dir).glob("part_*.parquet"))
        if existing_files:
            last_k = int(existing_files[-1].stem.split("_")[1])
            start_k = last_k + 1
            logger.warning(
                f"Found existing parquet files. Starting from part_{start_k}. Use --override for clean start."
            )
        else:
            start_k = 0
        return start_k


def mpi_manager(total_molecules: int, start_k: int, batch_size: int = 10000) -> None:
    """
    MPI manager process (rank 0) that coordinates processing tasks.

    Uses a blocking comm.recv loop instead of iprobe to atomically consume
    each message. This prevents phantom re-detection of unconsumed TAG_READY
    messages and ensures active_workers is always accurate.

    Args:
        total_molecules: Total number of molecules to process
        start_k: Starting parquet file index
        batch_size: Number of molecules per batch
    """
    num_batches = (total_molecules + batch_size - 1) // batch_size
    task_queue = deque(range(num_batches))
    total_molecules_processed = 0
    global_part_k = start_k
    active_workers = 0
    stopped_workers: set = set()

    logger.info(
        f"Starting MPI manager with {size - 1} workers, "
        f"{num_batches} batches ({batch_size} molecules each)"
    )

    while task_queue or active_workers > 0:
        status = MPI.Status()
        data = comm.recv(source=MPI_ANY_SOURCE, tag=MPI_ANY_TAG, status=status)
        sender = status.Get_source()
        tag = status.Get_tag()

        if tag == TAG_READY:
            if task_queue:
                batch_idx = task_queue.popleft()
                comm.send((batch_idx, global_part_k), dest=sender, tag=TAG_TASK)
                global_part_k += 1
                active_workers += 1
                logger.debug(f"Assigned batch {batch_idx} to worker {sender}")
            else:
                comm.send(None, dest=sender, tag=TAG_TASK)
                stopped_workers.add(sender)
                logger.debug(f"Sent stop signal to worker {sender} (no more tasks)")

        elif tag == TAG_DONE:
            mol_count, part_k = data
            active_workers -= 1
            total_molecules_processed += mol_count
            logger.info(
                f"Worker {sender} completed part_{part_k} ({mol_count} molecules)"
            )

        elif tag == TAG_ERROR:
            batch_idx, msg = data
            active_workers -= 1
            logger.warning(f"Worker {sender} failed on batch {batch_idx}: {msg}")

    for worker in range(1, size):
        if worker not in stopped_workers:
            comm.send(None, dest=worker, tag=TAG_TASK)
            logger.debug(f"Post-loop stop signal sent to worker {worker}")

    logger.info(
        f"MPI processing complete. Total: {total_molecules_processed} molecules"
    )


def mpi_worker(parquet_path: str, out_dir: str, batch_size: int = 10000) -> None:
    """
    MPI worker process that processes a specific batch of molecules.

    Args:
        parquet_path: Path to the source parquet file
        out_dir: Output directory for parquet files
        batch_size: Number of molecules per batch
    """
    logger.info(f"Worker {rank} starting")

    while True:
        comm.send(None, dest=0, tag=TAG_READY)
        task = comm.recv(source=0, tag=TAG_TASK)

        if task is None:
            logger.info(f"Worker {rank} received stop signal")
            break

        batch_idx, part_k = task
        logger.info(f"Worker {rank} processing batch: {batch_idx}")

        try:
            records = []
            for mol_id, smiles in get_molecule_batch(
                parquet_path, batch_size, batch_idx
            ):
                records.append({"id": mol_id, "SMILES": smiles})

            if records:
                output_path = f"{out_dir}/part_{part_k}.parquet"
                write_parquet(records, output_path)
                logger.info(
                    f"Worker {rank} wrote {output_path} ({len(records)} molecules)"
                )
                comm.send((len(records), part_k), dest=0, tag=TAG_DONE)
            else:
                logger.warning(f"Worker {rank} got no records from batch {batch_idx}")
                comm.send((0, part_k), dest=0, tag=TAG_DONE)

        except Exception as e:
            logger.error(f"Worker {rank} error on batch {batch_idx}: {e}")
            comm.send((batch_idx, str(e)), dest=0, tag=TAG_ERROR)


def main():
    parser = argparse.ArgumentParser(
        description="MolPILE SMILES Dataset Processor with MPI-based batching and automatic cleanup (MPI-only)"
    )
    parser.add_argument(
        "--out-dir",
        default="./data/raw/molpile",
        help="Output directory for parquet files",
    )
    parser.add_argument(
        "--override", action="store_true", help="Delete existing files and start fresh"
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10000,
        help="Number of molecules per batch (default: 10000)",
    )

    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    logger.info("Processing MolPILE dataset")

    if rank == 0:
        try:
            parquet_path = download_molpile_parquet()

            total_molecules = get_total_molecules(parquet_path)
            if total_molecules == 0:
                logger.error("No molecules found in parquet file")
                return 1

            start_k = setup_output_directory(args.out_dir, args.override)
            batch_size = args.batch_size

            comm.bcast(parquet_path, root=0)
            comm.bcast(total_molecules, root=0)
            comm.bcast(start_k, root=0)
            comm.bcast(batch_size, root=0)

        except Exception as e:
            logger.error(f"Rank 0 setup failed: {e}")
            return 1
    else:
        parquet_path = comm.bcast(None, root=0)
        total_molecules = comm.bcast(None, root=0)
        start_k = comm.bcast(None, root=0)
        batch_size = comm.bcast(None, root=0)

    if rank == 0:
        mpi_manager(total_molecules, start_k, batch_size)

        if os.path.exists(TEMP_DIR):
            try:
                shutil.rmtree(TEMP_DIR)
                logger.info(f"Cleaned up temporary file: {TEMP_DIR}")
            except Exception as e:
                logger.warning(f"Failed to clean up temporary file {TEMP_DIR}: {e}")
    else:
        mpi_worker(parquet_path, args.out_dir, batch_size)

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
