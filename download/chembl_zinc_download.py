import argparse
import os
import time
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from mpi4py import MPI
from tqdm import tqdm

BASE_DATA_DIR = "data"

# ── ZINC ──────────────────────────────────────────────────────────────────────

def get_zinc_tranche_urls(urls_file="download/zinc_urls.txt"):
    """Read ZINC tranche URLs from file."""
    with open(urls_file, "r") as f:
        urls = [line.strip() for line in f if line.strip()]
    return urls


def download_zinc_file(url, dest_folder):
    """Download a single ZINC .smi file with retry logic, headers, and throttling."""

    filename = url.split("/")[-1]
    dest_path = os.path.join(dest_folder, filename)

    if os.path.exists(dest_path):
        return dest_path

    # Force HTTPS (ZINC blocks many HTTP requests)
    url = url.replace("http://", "https://")

    session = requests.Session()

    retry = Retry(
        total=8,
        backoff_factor=3,
        status_forcelist=[403, 429, 500, 502, 503, 504],
        allowed_methods=["GET"]
    )

    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)

    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36",
        "Accept": "text/plain, */*",
        "Connection": "keep-alive"
    }

    try:
        response = session.get(url, headers=headers, timeout=120)

        if response.status_code == 200:
            with open(dest_path, "w") as f:
                f.write(response.text)

            # Light throttling to avoid cluster burst behavior
            time.sleep(0.8)
            return dest_path

        else:
            print(f"[Rank {MPI.COMM_WORLD.Get_rank()}] Status {response.status_code} for {url}")
            return None

    except Exception as e:
        print(f"[Rank {MPI.COMM_WORLD.Get_rank()}] Failed {url}: {e}")
        return None

    finally:
        time.sleep(1.0)  # extra cooling delay per request

def process_zinc_smi(file_path):
    """Parse a .smi file and return list of (zinc_id, smiles) tuples."""
    rows = []
    if file_path is None or not os.path.exists(file_path):
        return rows
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                smiles, zinc_id = line.split()
                if zinc_id == "zinc_id":
                    continue
                rows.append((zinc_id, smiles))
            except ValueError:
                pass
    return rows


# ── ChEMBL ────────────────────────────────────────────────────────────────────

def get_chembl_chunk_ranges(total, chunk_size=10000):
    """Return (offset, limit) pairs that cover all ChEMBL molecules."""
    return [(offset, chunk_size) for offset in range(0, total, chunk_size)]


def fetch_chembl_total():
    """Query ChEMBL REST API for the total molecule count."""
    url = "https://www.ebi.ac.uk/chembl/api/data/molecule?format=json&limit=1"
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    return response.json()["page_meta"]["total_count"]


def fetch_chembl_chunk(offset, limit):
    """Fetch one page of ChEMBL molecules; return list of (chembl_id, smiles)."""
    url = (
        f"https://www.ebi.ac.uk/chembl/api/data/molecule"
        f"?format=json&limit={limit}&offset={offset}"
    )
    session = requests.Session()
    retry = Retry(
        total=5,
        backoff_factor=2,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)

    response = session.get(url, timeout=120)
    response.raise_for_status()
    rows = []
    for mol in response.json().get("molecules", []):
        cid = mol.get("molecule_chembl_id", "")
        struct = mol.get("molecule_structures") or {}
        smiles = struct.get("canonical_smiles", "")
        if cid and smiles:
            rows.append((cid, smiles))
    return rows


# ── MPI helpers ───────────────────────────────────────────────────────────────

def scatter_chunks(comm, all_chunks):
    """Rank 0 scatters chunks across all ranks."""
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        split = [all_chunks[i::size] for i in range(size)]
    else:
        split = None

    my_chunks = comm.scatter(split, root=0)
    return my_chunks


def gather_and_write(comm, my_rows, output_path, header):
    """Gather all rows to rank 0 and write the final parquet file."""
    all_rows = comm.gather(my_rows, root=0)

    if comm.Get_rank() == 0:
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        df = pd.DataFrame(
            [row for rank_rows in all_rows for row in rank_rows],
            columns=header
        )
        df.to_parquet(output_path, index=False)
        print(f"\nWrote {len(df)} rows to {output_path}")


# ── ZINC main ─────────────────────────────────────────────────────────────────

def run_zinc(args, comm):
    rank = comm.Get_rank()

    if rank == 0:
        all_urls = get_zinc_tranche_urls(args.zinc_urls_file)
        print(f"[ZINC] Total URLs loaded: {len(all_urls)}")
        zinc_folder = os.path.join(BASE_DATA_DIR, "zinc_smi")
        os.makedirs(zinc_folder, exist_ok=True)
    else:
        all_urls = None
        zinc_folder = None

    all_urls    = comm.bcast(all_urls,    root=0)
    zinc_folder = comm.bcast(zinc_folder, root=0)

    my_urls = scatter_chunks(comm, all_urls)
    print(f"[Rank {rank}] Assigned {len(my_urls)} URLs")

    my_rows = []
    for url in tqdm(my_urls, desc=f"[Rank {rank}] ZINC", unit="file"):
        path = download_zinc_file(url, zinc_folder)
        my_rows.extend(process_zinc_smi(path))

    output_path = os.path.join(BASE_DATA_DIR, args.zinc_output)
    gather_and_write(comm, my_rows, output_path, ["zinc_id", "smiles"])


# ── ChEMBL main ───────────────────────────────────────────────────────────────

def run_chembl(args, comm):
    rank = comm.Get_rank()

    if rank == 0:
        total = fetch_chembl_total()
        print(f"[ChEMBL] Total molecules: {total}")
        all_chunks = get_chembl_chunk_ranges(total, chunk_size=args.chembl_chunk_size)
        print(f"[ChEMBL] Total chunks: {len(all_chunks)}")
    else:
        all_chunks = None

    rank = comm.Get_rank()
    
    all_chunks = comm.bcast(all_chunks, root=0)
    print(f"[Rank {rank}] Starting with throttle mode enabled")
    my_chunks  = scatter_chunks(comm, all_chunks)

    my_rows = []
    for offset, limit in tqdm(my_chunks, desc=f"[Rank {rank}] ChEMBL", unit="chunk"):
        try:
            my_rows.extend(fetch_chembl_chunk(offset, limit))
        except Exception as e:
            print(f"[Rank {rank}] Error at offset {offset}: {e}")

    output_path = os.path.join(BASE_DATA_DIR, args.chembl_output)
    gather_and_write(comm, my_rows, output_path, ["chembl_id", "smiles"])


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Parallel ZINC + ChEMBL downloader using MPI"
    )
    parser.add_argument("--zinc",              action="store_true", help="Download ZINC")
    parser.add_argument("--chembl",            action="store_true", help="Download ChEMBL")
    parser.add_argument("--zinc-urls-file",    default="download/zinc_urls.txt",
                                               help="Path to file containing ZINC URLs")
    parser.add_argument("--zinc-output",       default="zinc_smiles.parquet")
    parser.add_argument("--chembl-output",     default="chembl_smiles.parquet")
    parser.add_argument("--chembl-chunk-size", type=int, default=10000)
    args = parser.parse_args()

    comm = MPI.COMM_WORLD

    if args.zinc:
        run_zinc(args, comm)

    if args.chembl:
        run_chembl(args, comm)

    if not args.zinc and not args.chembl:
        if comm.Get_rank() == 0:
            print("Specify --zinc and/or --chembl")


if __name__ == "__main__":
    main()