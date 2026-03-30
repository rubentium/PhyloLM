"""
preprocess_memmaps.py

Reads FASTA/Newick pairs, tokenises them with the ESM2 tokenizer, and writes
the results as memory-mapped binary files under a structured output directory.

Output layout:
    <output_dir>/
        train/
            alignments.dat   np.memmap  int8    (N_train, R, C)
            trees.dat        np.memmap  int16   (N_train, P)   [bfloat16 bits]
            meta.json
        val/
            alignments.dat   np.memmap  int8    (N_val, R, C)
            trees.dat        np.memmap  int16   (N_val, P)     [bfloat16 bits]
            meta.json

Alignment tokens fit in int8 because the ESM2 vocab has only 33 entries.
Tree distances are bfloat16 stored as int16 raw bits (same 2-byte footprint,
no extra dependencies). Use `tensor.view(torch.bfloat16)` after loading.

P = R*(R-1)/2  (1 225 for R=50)

Usage:
    python preprocess_memmaps.py \\
        --train_alignment_dir /path/to/train/alignments \\
        --train_tree_dir      /path/to/train/trees      \\
        --val_alignment_dir   /path/to/val/alignments   \\
        --val_tree_dir        /path/to/val/trees        \\
        [--output_dir         LG_GC_memmaps]            \\
        [--num_workers        <int, default os.cpu_count()>]
"""

import os
import sys
import json
import logging
import argparse
import numpy as np
import torch
import torch.multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

# make sure the package root is on the path so we can import model.data
sys.path.insert(0, os.path.dirname(__file__))
from .data import discover_pairs, _preprocess_one, Tokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

try:
    mp.set_sharing_strategy('file_system')
except RuntimeError:
    pass

def _distances_to_int16(distances: np.ndarray) -> np.ndarray:
    """
    Cast a float32 distance array to bfloat16, then reinterpret the raw bits
    as int16 so numpy can store them in a memmap without knowing about bfloat16.
    """
    return torch.from_numpy(distances).to(torch.bfloat16).view(torch.int16).numpy()


def _write_split(
    alignment_dir: str,
    tree_dir: str,
    out_dir: str,
    tokenizer: Tokenizer,
    num_workers: int,
) -> dict:
    os.makedirs(out_dir, exist_ok=True)

    pairs = discover_pairs(alignment_dir, tree_dir)
    n_samples = len(pairs)
    if n_samples == 0:
        raise RuntimeError(f"No matched pairs in {alignment_dir}")

    logger.info("[%s] Peeking at first sample to determine shapes...", out_dir)
    first_seqs, first_dist = _preprocess_one(pairs[0])
    first_ids = tokenizer.encode(first_seqs)
    R, C = first_ids.shape
    P = first_dist.shape[0]

    align_path = os.path.join(out_dir, "alignments.dat")
    trees_path  = os.path.join(out_dir, "trees.dat")
    align_mm = np.memmap(align_path, dtype=np.int8,  mode="w+", shape=(n_samples, R, C))
    trees_mm  = np.memmap(trees_path, dtype=np.int16, mode="w+", shape=(n_samples, P))

    logger.info("[%s] Preprocessing %d samples with %d workers...", out_dir, n_samples, num_workers)
    
    with ProcessPoolExecutor(max_workers=num_workers) as pool:
        chunksize = max(1, n_samples // (num_workers * 4))
        results = pool.map(_preprocess_one, pairs, chunksize=chunksize)

        for i, (sequences, distances) in enumerate(results):
            ids = tokenizer.encode(sequences)
            
            align_mm[i] = ids.numpy().astype(np.int8)
            trees_mm[i] = _distances_to_int16(distances)

            if (i + 1) % 1000 == 0:
                logger.info("[%s] Written %d / %d samples", out_dir, i + 1, n_samples)

    align_mm.flush()
    trees_mm.flush()
    del align_mm, trees_mm

    meta = {
        "num_samples": n_samples, "num_seqs": R, "seq_len": C, "num_pairs": P,
        "alignments": {"file": "alignments.dat", "dtype": "int8", "shape": [n_samples, R, C]},
        "trees": {"file": "trees.dat", "dtype": "int16", "shape": [n_samples, P]}
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    return meta

def parse_args():
    p = argparse.ArgumentParser(
        description="Preprocess FASTA/Newick pairs and write np.memmap binary files."
    )
    p.add_argument("--train_alignment_dir", required=True,
                   help="Directory containing {id}_50_tips.fasta training files")
    p.add_argument("--train_tree_dir",      required=True,
                   help="Directory containing {id}_50_tips.nwk training files")
    p.add_argument("--val_alignment_dir",   required=True,
                   help="Directory containing {id}_50_tips.fasta validation files")
    p.add_argument("--val_tree_dir",        required=True,
                   help="Directory containing {id}_50_tips.nwk validation files")
    p.add_argument("--output_dir", default="LG_GC_memmaps",
                   help="Root directory for output memmaps (default: LG_GC_memmaps)")
    return p.parse_args()


def main():
    args = parse_args()

    num_workers = len(os.sched_getaffinity(0))
    print(f"Using {num_workers} worker processes for preprocessing")
    logger.info("Using %d worker processes for preprocessing", num_workers)

    tokenizer = Tokenizer()
    logger.info("Tokenizer loaded (vocab size: %d)", len(tokenizer))

    for split, al_dir, tr_dir in [
        ("train", args.train_alignment_dir, args.train_tree_dir),
        ("val",   args.val_alignment_dir,   args.val_tree_dir),
    ]:
        out_dir = os.path.join(args.output_dir, split)
        logger.info("=== Processing split: %s ===", split)
        meta = _write_split(al_dir, tr_dir, out_dir, tokenizer, num_workers)
        logger.info(
            "Split %s complete — %d samples, alignments.dat %s int8, trees.dat %s int16(bf16 bits)",
            split, meta["num_samples"],
            meta["alignments"]["shape"],
            meta["trees"]["shape"],
        )

    logger.info("All splits done. Output at: %s", args.output_dir)


if __name__ == "__main__":
    main()
