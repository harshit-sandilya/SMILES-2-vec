#!/usr/bin/env python3
"""
ZINC Downloader — Bulk download SMILES from ZINC20 URI lists
and save as parquet files in data/raw directory.

This script requires MPI and must be run with mpirun/mpiexec.
Default source: zinc20.uri (publicly accessible)

# Run with default zinc20.uri file
mpirun -np 4 python download_zinc.py

# Specify custom source file
mpirun -np 4 python download_zinc.py --src custom.uri

# Override the exisitng data for the new one
mpirun -np 4 python download_zinc.py --override
"""

import argparse
import logging
import os
import sys
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


def stream_url(url: str) -> Iterator[Tuple[str, str]]:
    """
    Stream SMILES data from a ZINC URL and yield (zinc_id, smiles) tuples.

    Args:
        url: URL to download from

    Yields:
        Tuple of (zinc_id, smiles) for each molecule

    Raises:
        requests.exceptions.RequestException: On download errors
    """
    session = requests.Session()
    session.headers.update(
        {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    )
    response = session.get(url, stream=True, timeout=60)
    response.raise_for_status()

    lines = response.iter_lines()
    lines = (
        line.decode("utf-8", errors="replace") if isinstance(line, bytes) else line
        for line in lines
    )

    first = True
    for line in lines:
        if first:  # skip header
            first = False
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        smiles, zinc_id = parts[0], parts[1]
        yield zinc_id, smiles


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


def load_url_list(src_path: str) -> List[str]:
    """
    Load URL list from source file, filtering out comments and empty lines.

    Args:
        src_path: Path to .uri file

    Returns:
        List of URLs
    """
    url_list = []
    with open(src_path, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                url_list.append(line)
    return url_list


def setup_output_directory(out_dir: str, override: bool) -> int:
    """
    Set up output directory and determine starting point.

    Args:
        out_dir: Output directory path
        override: Whether to override existing files

    Returns:
        start_k: starting index of parquet part
    """
    # Create output directory if it doesn't exist
    os.makedirs(out_dir, exist_ok=True)

    if override:
        # Delete existing parquet files
        for f in Path(out_dir).glob("part_*.parquet"):
            f.unlink()
        return 0
    else:
        # Determine starting point from existing files
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


def mpi_manager(url_list: List[str], src_path: str, start_k: int) -> None:
    """
    MPI manager process (rank 0) that coordinates download tasks.

    Args:
        url_list: List of URLs to download
        src_path: Source URI file path
        start_k: Starting parquet file index
    """

    task_queue = deque(url_list)
    total_molecules = 0
    global_part_k = start_k
    active_workers = 0
    url_cursor = 0
    stopped_workers: set = set()

    logger.info(f"Starting MPI manager with {size - 1} workers, {len(task_queue)} URLs")

    # Main manager loop
    while task_queue or active_workers > 0:
        status = MPI.Status()
        data = comm.recv(source=MPI_ANY_SOURCE, tag=MPI_ANY_TAG, status=status)
        sender = status.Get_source()
        tag = status.Get_tag()

        if tag == TAG_READY:
            if task_queue:
                url = task_queue.popleft()
                comm.send((url, global_part_k), dest=sender, tag=TAG_TASK)
                global_part_k += 1
                active_workers += 1
                logger.debug(f"Assigned URL {url_cursor} to worker {sender}")
                url_cursor += 1
            else:
                comm.send(None, dest=sender, tag=TAG_TASK)
                stopped_workers.add(sender)
                logger.debug(f"Sent stop signal to worker {sender} (no more tasks)")

        elif tag == TAG_DONE:
            mol_count, part_k = data
            active_workers -= 1
            total_molecules += mol_count
            logger.info(
                f"Worker {sender} completed part_{part_k} ({mol_count} molecules)"
            )

        elif tag == TAG_ERROR:
            url, msg = data
            active_workers -= 1
            logger.warning(f"Worker {sender} failed on {url}: {msg}")

    # Send stop signals to any remaining workers
    for worker in range(1, size):
        if worker not in stopped_workers:
            comm.send(None, dest=worker, tag=TAG_TASK)
            logger.debug(f"Post-loop stop signal sent to worker {worker}")

    logger.info(f"MPI download complete. Total: {total_molecules} molecules")


def mpi_worker(out_dir: str) -> None:
    """
    MPI worker process that downloads data from assigned URLs.

    Args:
        out_dir: Output directory for parquet files
    """
    logger.info(f"Worker {rank} starting")

    while True:
        # Signal readiness
        comm.send(None, dest=0, tag=TAG_READY)

        # Receive task
        task = comm.recv(source=0, tag=TAG_TASK)

        if task is None:
            # Stop signal received
            logger.info(f"Worker {rank} received stop signal")
            break

        url, part_k = task
        logger.info(f"Worker {rank} processing URL: {url}")

        try:
            records = []
            for zinc_id, smiles in stream_url(url):
                records.append({"id": zinc_id, "SMILES": smiles})

            if records:
                output_path = f"{out_dir}/part_{part_k}.parquet"
                write_parquet(records, output_path)
                logger.info(
                    f"Worker {rank} wrote {output_path} ({len(records)} molecules)"
                )
                comm.send((len(records), part_k), dest=0, tag=TAG_DONE)
            else:
                logger.warning(f"Worker {rank} got no records from {url}")
                comm.send((0, part_k), dest=0, tag=TAG_DONE)

        except Exception as e:
            logger.error(f"Worker {rank} error on {url}: {e}")
            comm.send((url, str(e)), dest=0, tag=TAG_ERROR)


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="ZINC20 SMILES Bulk Downloader (MPI-only)"
    )
    parser.add_argument(
        "--src",
        default="./download/zinc20.uri",
        help="Path to .uri file (one URL per line). Default: zinc20.uri",
    )
    parser.add_argument(
        "--out-dir",
        default="./data/raw/zinc",
        help="Output directory for parquet files",
    )
    parser.add_argument(
        "--override", action="store_true", help="Delete existing files and start fresh"
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    logger.info(f"Using source file: {args.src}")

    # Only rank 0 loads the URL list
    if rank == 0:
        # Load URL list
        url_list = load_url_list(args.src)
        if not url_list:
            logger.error("No URLs found in source file")
            return 1

        # Set up output directory and determine starting point
        start_k = setup_output_directory(args.out_dir, args.override)

        # Broadcast setup information to workers
        comm.bcast(start_k, root=0)
        comm.bcast(url_list, root=0)
    else:
        # Worker processes wait for broadcast
        start_k = comm.bcast(None, root=0)
        url_list = comm.bcast(None, root=0)

    # Execute MPI download
    if rank == 0:
        mpi_manager(url_list, args.src, start_k)
    else:
        mpi_worker(args.out_dir)

    return 0


if __name__ == "__main__":
    sys.exit(main())
