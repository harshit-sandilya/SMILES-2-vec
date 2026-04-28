#!/usr/bin/env python3
"""
build_dataset.py  —  Spark molecular-data pipeline
====================================================
Reads raw parquets from data/raw/{molpile,zinc,gdb13},
canonicalises SMILES with RDKit, deduplicates by InChIKey
(priority: molpile > zinc > gdb13), filters by heavy-atom
count, and writes fixed-size shards to data/processed/.

Run via the companion SLURM script (process_dataset.sh),
or directly:
    python build_dataset.py
    python build_dataset.py --max-atoms 96 --shard-size 50000
    python build_dataset.py --raw-dir /scratch/data/raw --out-dir /scratch/data/processed
"""

import argparse
import logging
import sys
from functools import reduce
from pathlib import Path

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TOP-LEVEL CONSTANTS  (edit here; all are also CLI-overridable)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MAX_ATOMS: int = 64  # heavy-atom upper bound (inclusive)
MIN_ATOMS: int = 3  # drop trivial fragments (H2O, CO2 …)
SHARD_SIZE: int = 10_000  # rows per output parquet shard
SPARK_CORES: int = 224  # local[N] — set to match SLURM cpus-per-task

RAW_DIR = "data/raw"
OUT_DIR = "data/processed"

# Source union order; lower priority number wins dedup tie on InChIKey
SOURCES = [
    ("molpile", 0),  # highest quality — curated, deduplicated upstream
    ("zinc", 1),  # purchasable, drug-like
    ("gdb13", 2),  # theoretical — lowest priority
]

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LOGGING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SPARK SESSION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def create_spark(cores: int, driver_mem: str, local_dir: str):
    from pyspark.sql import SparkSession

    # Rule of thumb for shuffle partitions: 3–4× number of cores
    shuffle_parts = max(400, cores * 4)

    spark = (
        SparkSession.builder.appName("molecular-dataset-builder")
        .master(f"local[{cores}]")
        # ── Memory ──────────────────────────────────────────────────────
        # In local mode the driver IS the executor.
        # RDKit (C++ via pandas_udf) allocates native memory on top of JVM.
        # Leave ~15 % of the node for OS + RDKit native heap (done in .sh).
        .config("spark.driver.memory", driver_mem)
        .config("spark.driver.maxResultSize", "16g")
        # ── Shuffle / AQE ────────────────────────────────────────────────
        .config("spark.sql.shuffle.partitions", str(shuffle_parts))
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        .config("spark.sql.adaptive.skewJoin.enabled", "true")
        # ── Arrow ────────────────────────────────────────────────────────
        # Arrow is required for pandas_udf (struct-returning)
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.sql.execution.arrow.maxRecordsPerBatch", "8192")
        # ── I/O ──────────────────────────────────────────────────────────
        .config("spark.local.dir", local_dir)
        .config("spark.sql.parquet.compression.codec", "snappy")
        .config("spark.sql.files.maxPartitionBytes", "256m")
        .config("spark.sql.files.openCostInBytes", "8m")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MOLECULE PROCESSING UDF
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def make_processor_udf():
    """
    Returns a pandas_udf that maps a SMILES Series → struct:
        canonical_smiles : str   (RDKit canonical, isomeric)
        inchikey         : str   (27-char InChIKey for dedup)
        num_atoms        : int   (heavy atom count only)
        valid            : bool

    Everything is done in one RDKit pass per molecule:
        1. Parse SMILES
        2. Pick largest fragment (drops salts / solvents)
        3. Neutralise formal charges
        4. Canonical SMILES  +  InChIKey  +  heavy-atom count

    Imports are intentionally inside the closure so this UDF can be
    serialised and shipped to remote executors without issue.
    """
    import pandas as pd
    from pyspark.sql.functions import pandas_udf
    from pyspark.sql.types import (
        BooleanType,
        IntegerType,
        StringType,
        StructField,
        StructType,
    )

    _schema = StructType(
        [
            StructField("canonical_smiles", StringType(), nullable=True),
            StructField("inchikey", StringType(), nullable=True),
            StructField("num_atoms", IntegerType(), nullable=True),
            StructField("valid", BooleanType(), nullable=False),
        ]
    )

    @pandas_udf(_schema)
    def _process(smiles_series: pd.Series) -> pd.DataFrame:
        # ── heavy imports inside UDF (once per Arrow batch) ─────────────
        from rdkit import Chem, RDLogger
        from rdkit.Chem.inchi import MolToInchiKey
        from rdkit.Chem.MolStandardize import rdMolStandardize

        RDLogger.DisableLog("rdApp.*")  # silence noisy RDKit warnings

        frag_chooser = rdMolStandardize.LargestFragmentChooser()
        uncharger = rdMolStandardize.Uncharger()

        rows = []
        for smi in smiles_series:
            try:
                mol = Chem.MolFromSmiles(smi)
                if mol is None:
                    rows.append((None, None, None, False))
                    continue

                mol = frag_chooser.choose(mol)  # largest fragment
                mol = uncharger.uncharge(mol)  # neutralise charges
                canonical = Chem.MolToSmiles(mol, isomericSmiles=True)
                inchikey = MolToInchiKey(mol)
                num_atoms = mol.GetNumHeavyAtoms()

                rows.append((canonical, inchikey, num_atoms, True))
            except Exception:
                rows.append((None, None, None, False))

        return pd.DataFrame(
            rows,
            columns=["canonical_smiles", "inchikey", "num_atoms", "valid"],
        )

    return _process


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LOAD RAW SOURCES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def load_raw(spark, raw_dir: str):
    from pyspark.sql.functions import lit

    frames = []
    for name, priority in SOURCES:
        src = Path(raw_dir) / name
        if not src.exists():
            log.warning("  %-10s  ✗  not found at %s — skipping", name, src)
            continue

        df = (
            spark.read.parquet(str(src))
            .select("id", "SMILES")
            .withColumn("_priority", lit(priority))
        )
        n = df.count()
        log.info("  %-10s  ✓  %s rows", name, f"{n:,}")
        frames.append(df)

    if not frames:
        log.error("No source data found under %s", raw_dir)
        sys.exit(1)

    from pyspark.sql import DataFrame

    return reduce(DataFrame.union, frames)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CANONICALISE + FILTER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def canonicalise_and_filter(df, min_atoms: int, max_atoms: int):
    from pyspark.sql.functions import col

    processor = make_processor_udf()

    return (
        df.withColumn("_mol", processor(col("SMILES")))
        .filter(col("_mol.valid") == True)  # noqa: E712
        .filter(col("_mol.inchikey").isNotNull())
        .filter(col("_mol.num_atoms") >= min_atoms)
        .filter(col("_mol.num_atoms") <= max_atoms)
        .select(
            col("id"),
            col("_mol.canonical_smiles").alias("SMILES"),
            col("_mol.inchikey").alias("_inchikey"),
            col("_priority"),
        )
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DEDUPLICATION  (O(n) hash-aggregate, no sort)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def deduplicate(df):
    """
    Priority-aware global deduplication on InChIKey.

    Strategy: groupBy(inchikey) + min(struct(_priority, id, SMILES))
    ─────────────────────────────────────────────────────────────────
    struct comparison in Spark is lexicographic, so the struct with
    the smallest _priority value always wins:

        (0, "MOLPILE_...", "CCO") < (1, "ZINC...", "CCO")
                                  < (2, "GDB13_...", "CCO")

    This is a pure hash-aggregate (no sort / Window needed) → O(n).
    The bloom filter is Spark's internal hash-table for groupBy;
    no external bloom library is required at this scale.
    """
    from pyspark.sql.functions import col, struct
    from pyspark.sql.functions import min as spark_min

    return (
        df.groupBy("_inchikey")
        .agg(spark_min(struct("_priority", "id", "SMILES")).alias("_best"))
        .select(
            col("_best.id").alias("id"),
            col("_best.SMILES").alias("SMILES"),
        )
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# WRITE SHARDS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def write_shards(df, out_dir: str, shard_size: int, cores: int):
    """
    Write parquet shards of exactly `shard_size` rows each.

    maxRecordsPerFile is the Spark-native knob for output shard sizing.
    We repartition first to:
      (a) distribute write load across all cores
      (b) keep post-coalesce file count predictable
    """
    n_write_partitions = max(cores, 400)

    (
        df.repartition(n_write_partitions)
        .write.mode("overwrite")
        .option("maxRecordsPerFile", str(shard_size))
        .parquet(out_dir)
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def main():
    p = argparse.ArgumentParser(
        description="Build a deduplicated, canonical SMILES dataset from raw parquets",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--max-atoms",
        type=int,
        default=MAX_ATOMS,
        help="Heavy-atom upper bound (inclusive)",
    )
    p.add_argument(
        "--min-atoms",
        type=int,
        default=MIN_ATOMS,
        help="Heavy-atom lower bound (inclusive)",
    )
    p.add_argument(
        "--shard-size",
        type=int,
        default=SHARD_SIZE,
        help="Rows per output parquet shard",
    )
    p.add_argument(
        "--cores", type=int, default=SPARK_CORES, help="Spark local[N] thread count"
    )
    p.add_argument(
        "--driver-memory",
        default="400g",
        help="Spark JVM driver memory (set by process_dataset.sh)",
    )
    p.add_argument(
        "--raw-dir",
        default=RAW_DIR,
        help="Root directory containing molpile/, zinc/, gdb13/",
    )
    p.add_argument(
        "--out-dir", default=OUT_DIR, help="Output directory for processed shards"
    )
    p.add_argument(
        "--local-dir",
        default=None,
        help="Spark scratch dir for shuffle spills (defaults to $TMPDIR or /tmp)",
    )
    args = p.parse_args()

    # ── Scratch dir for Spark spills ────────────────────────────────────
    import os

    local_dir = args.local_dir or os.environ.get("TMPDIR") or "/tmp"
    local_dir = str(Path(local_dir) / f"spark_mol_{os.getpid()}")
    Path(local_dir).mkdir(parents=True, exist_ok=True)

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    log.info("━" * 64)
    log.info("  Molecular Dataset Builder")
    log.info("  raw dir      : %s", args.raw_dir)
    log.info("  out dir      : %s", args.out_dir)
    log.info("  atom range   : [%d, %d]", args.min_atoms, args.max_atoms)
    log.info("  shard size   : %s rows", f"{args.shard_size:,}")
    log.info("  cores        : %d", args.cores)
    log.info("  driver mem   : %s", args.driver_memory)
    log.info("  spark tmp    : %s", local_dir)
    log.info("━" * 64)

    # ── Spark ────────────────────────────────────────────────────────────
    spark = create_spark(args.cores, args.driver_memory, local_dir)

    # ── Stage 1 — load ───────────────────────────────────────────────────
    log.info("[1/4] Loading raw sources …")
    raw = load_raw(spark, args.raw_dir)

    # ── Stage 2 — canonicalise + filter ──────────────────────────────────
    log.info(
        "[2/4] Canonicalising SMILES and filtering (atoms: %d–%d) …",
        args.min_atoms,
        args.max_atoms,
    )
    filtered = canonicalise_and_filter(raw, args.min_atoms, args.max_atoms)

    # ── Stage 3 — deduplicate ────────────────────────────────────────────
    log.info("[3/4] Deduplicating by InChIKey (priority: molpile > zinc > gdb13) …")
    unique = deduplicate(filtered)

    # ── Stage 4 — write ──────────────────────────────────────────────────
    log.info(
        "[4/4] Writing %s-row shards to %s …", f"{args.shard_size:,}", args.out_dir
    )
    write_shards(unique, args.out_dir, args.shard_size, args.cores)

    # ── Summary ──────────────────────────────────────────────────────────
    n_out = spark.read.parquet(args.out_dir).count()
    n_files = len(list(Path(args.out_dir).glob("*.parquet")))
    log.info("━" * 64)
    log.info("  Finished.")
    log.info("  Unique molecules : %s", f"{n_out:,}")
    log.info("  Output shards    : %s files", f"{n_files:,}")
    log.info("━" * 64)

    spark.stop()


if __name__ == "__main__":
    main()
