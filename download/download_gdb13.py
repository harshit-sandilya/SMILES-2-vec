#!/usr/bin/env python3
"""
GDB13 Downloader — Download GDB13 SMILES archive from gdb.unibe.ch,
extract it, and save as parquet files in data/raw directory.

This script requires MPI and must be run with mpirun/mpiexec.
Source: https://gdb.unibe.ch/downloads/

# Run with full dataset (default)
mpirun -np 4 python download_gdb13.py

# Override existing data
mpirun -np 4 python download_gdb13.py --override

# Custom output directory
mpirun -np 4 python download_gdb13.py --out-dir ./custom/path

# Custom batch size (molecules per file)
mpirun -np 4 python download_gdb13.py --batch-size 5000
"""

import argparse
import logging
import os
import shutil
import sys
import tarfile
from collections import deque
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

import pyarrow as pa
import pyarrow.parquet as pq
import requests
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
GDB13_FULL = "gdb13.tgz"


# ── Download & extraction ────────────────────────────────────────────────────


def download_gdb(file_name: str, destination: str) -> str:
    """
    Download a GDB13 archive from gdb.unibe.ch.

    Args:
        file_name:   Archive filename (e.g. "gdb13.tgz")
        destination: Local directory to save into

    Returns:
        Absolute path to the saved archive
    """
    os.makedirs(destination, exist_ok=True)
    out_file = os.path.join(destination, file_name)

    url = f"https://zenodo.org/record/7041051/files/{file_name}?download=1"
    session = requests.Session()

    with session.get(url, stream=True) as r:
        r.raise_for_status()
        with open(out_file, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)
    logger.info(f"Saved: {out_file}")
    return out_file


def extract_archive(archive_path: str, extract_dir: str) -> List[str]:
    """
    Extract a .tgz archive and return a sorted list of .smi file paths.

    Args:
        archive_path: Path to the .tgz archive
        extract_dir:  Directory to extract into

    Returns:
        Sorted list of absolute .smi file paths

    Raises:
        RuntimeError: If no .smi files are found after extraction
    """
    logger.info(f"Extracting {archive_path} → {extract_dir}")
    os.makedirs(extract_dir, exist_ok=True)
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(path=extract_dir)

    smi_files = sorted(str(p) for p in Path(extract_dir).rglob("*.smi"))
    if not smi_files:
        raise RuntimeError(f"No .smi files found after extracting {archive_path}")
    logger.info(f"Found {len(smi_files)} .smi file(s)")
    return smi_files


# ── Task list construction ───────────────────────────────────────────────────


def count_lines(path: str) -> int:
    """
    Count lines via binary buffer reads (4 MB chunks).
    Faster than Python-level readline() for large .smi files.
    """
    count = 0
    with open(path, "rb") as f:
        while chunk := f.read(4 * 1024 * 1024):
            count += chunk.count(b"\n")
    return count


def build_flat_tasks(smi_files: List[str], batch_size: int) -> List[Tuple[str, int]]:
    """
    Build a flat list of (smi_path, local_batch_index) tuples.

    This allows the manager to stay structurally identical to the MolPILE
    script — it simply dispatches integer indices 0..N-1 into this list,
    without knowing anything about individual files or their sizes.

    Args:
        smi_files:  Sorted list of .smi file paths
        batch_size: Lines per output parquet file

    Returns:
        Flat list of (smi_path, local_batch_idx) in processing order
    """
    flat_tasks: List[Tuple[str, int]] = []
    for smi_path in smi_files:
        n_lines = count_lines(smi_path)
        if n_lines == 0:
            logger.warning(f"Skipping empty file: {smi_path}")
            continue
        n_batches = (n_lines + batch_size - 1) // batch_size
        for local_idx in range(n_batches):
            flat_tasks.append((smi_path, local_idx))
        logger.info(
            f"  {Path(smi_path).name}: {n_lines:,} lines → {n_batches} batch(es)"
        )
    return flat_tasks


# ── Molecule reading ─────────────────────────────────────────────────────────


def get_molecule_batch(
    smi_path: str, batch_size: int, batch_index: int
) -> Iterator[Tuple[str, str]]:
    """
    Yield (id, SMILES) tuples for a line-range batch inside a .smi file.

    GDB13 .smi files contain one SMILES per line with no header and no ID
    column. IDs are generated as:
        GDB13-{file_stem}-{zero_padded_line_number}
    e.g. GDB13-gdb13.07-0000042137

    Args:
        smi_path:    Path to the .smi file
        batch_size:  Lines per batch
        batch_index: Which batch to read (0-indexed, local to this file)

    Yields:
        (mol_id, smiles) tuples

    Raises:
        Exception: On file read errors
    """
    stem = Path(smi_path).stem
    start_line = batch_index * batch_size
    end_line = start_line + batch_size

    try:
        with open(smi_path, "r", encoding="utf-8", errors="replace") as f:
            for lineno, raw in enumerate(f):
                if lineno < start_line:
                    continue
                if lineno >= end_line:
                    break
                smiles = raw.strip()
                if not smiles or smiles.startswith("#"):
                    continue
                mol_id = f"GDB13-{stem}-{lineno:010d}"
                yield mol_id, smiles
    except Exception as e:
        logger.error(f"Error reading batch {batch_index} from {smi_path}: {e}")
        raise


# ── Parquet I/O ──────────────────────────────────────────────────────────────


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


# ── Directory setup ──────────────────────────────────────────────────────────


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


# ── MPI manager ──────────────────────────────────────────────────────────────


def mpi_manager(total_batches: int, start_k: int, batch_size: int = 10000) -> None:
    """
    MPI manager process (rank 0) that coordinates processing tasks.

    Identical to the MolPILE manager — dispatches integer indices 0..total_batches-1.
    Workers resolve (smi_path, local_batch_idx) from the broadcasted flat_tasks list.

    Args:
        total_batches: Total number of (file, batch) pairs to process
        start_k: Starting parquet file index
        batch_size: Number of molecules per batch
    """
    task_queue = deque(range(total_batches))
    total_molecules_processed = 0
    global_part_k = start_k
    active_workers = 0
    stopped_workers: set = set()

    logger.info(
        f"Starting MPI manager with {size - 1} workers, "
        f"{total_batches} batches ({batch_size} molecules each)"
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


# ── MPI worker ───────────────────────────────────────────────────────────────


def mpi_worker(
    flat_tasks: List[Tuple[str, int]], out_dir: str, batch_size: int = 10000
) -> None:
    """
    MPI worker process that processes a specific batch of molecules.

    Identical to the MolPILE worker except that the received batch_idx is
    used to look up (smi_path, local_batch_idx) from flat_tasks before
    calling get_molecule_batch.

    Args:
        flat_tasks: Flat list of (smi_path, local_batch_idx) broadcast from rank 0
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
        smi_path, local_batch_idx = flat_tasks[batch_idx]
        logger.info(
            f"Worker {rank} processing batch {batch_idx} "
            f"({Path(smi_path).name}[{local_batch_idx}])"
        )

        try:
            records = []
            for mol_id, smiles in get_molecule_batch(
                smi_path, batch_size, local_batch_idx
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


# ── Entry point ──────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="GDB13 SMILES Dataset Processor with MPI-based batching (MPI-only)"
    )
    parser.add_argument(
        "--out-dir",
        default="./data/raw/gdb13",
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

    logger.info("Processing GDB13 dataset")

    if rank == 0:
        try:
            file_name = GDB13_FULL
            archive_path = download_gdb(file_name, TEMP_DIR)

            extract_dir = os.path.join(TEMP_DIR, "extracted")
            smi_files = extract_archive(archive_path, extract_dir)

            flat_tasks = build_flat_tasks(smi_files, args.batch_size)
            total_batches = len(flat_tasks)

            if total_batches == 0:
                logger.error("No batches found — all .smi files may be empty")
                return 1

            start_k = setup_output_directory(args.out_dir, args.override)
            batch_size = args.batch_size

            comm.bcast(flat_tasks, root=0)
            comm.bcast(total_batches, root=0)
            comm.bcast(start_k, root=0)
            comm.bcast(batch_size, root=0)

        except Exception as e:
            logger.error(f"Rank 0 setup failed: {e}")
            return 1
    else:
        flat_tasks = comm.bcast(None, root=0)
        total_batches = comm.bcast(None, root=0)
        start_k = comm.bcast(None, root=0)
        batch_size = comm.bcast(None, root=0)

    if rank == 0:
        mpi_manager(total_batches, start_k, batch_size)

        if os.path.exists(TEMP_DIR):
            try:
                shutil.rmtree(TEMP_DIR)
                logger.info(f"Cleaned up temporary dir: {TEMP_DIR}")
            except Exception as e:
                logger.warning(f"Failed to clean up temporary dir {TEMP_DIR}: {e}")
    else:
        mpi_worker(flat_tasks, args.out_dir, batch_size)

    return 0


if __name__ == "__main__":
    sys.exit(main())
