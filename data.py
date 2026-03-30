import os
import re
import numpy as np
import torch
import logging
import dendropy
from itertools import combinations
from concurrent.futures import ProcessPoolExecutor
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)

# tokenizer

class Tokenizer:
    """
    stub tokenizer for amino acid sequences
    when implemented, should map each of the 20 standard amino acids to 0-19
    and the gap character '-' to 20
    arguments:
        sequences: list of R equal-length aligned amino acid strings
    returns:
        LongTensor of shape (R, C) where C is the alignment length
    """
    
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")

    def encode(self, sequences):
        tokenized = self.tokenizer(sequences, padding=True, truncation=True, return_tensors="pt")
        return tokenized.input_ids
    
    def __len__(self):
        return len(self.tokenizer)


# file discovery

def discover_pairs(alignment_dir, tree_dir):
    """
    finds matching fasta/newick file pairs by extracting {id} from filenames
    of the form {id}_50.fasta and {id}_50.nwck — any unpaired files are dropped with a warning
    returns a sorted list of (fasta_path, nwck_path) tuples
    """
    fasta_pattern = re.compile(r"^(.+)_50_tips\.fasta$")
    nwck_pattern = re.compile(r"^(.+)_50_tips\.nwk$")

    fasta_ids = {}
    for fname in os.listdir(alignment_dir):
        m = fasta_pattern.match(fname)
        if m:
            fasta_ids[m.group(1)] = os.path.join(alignment_dir, fname)

    nwck_ids = {}
    for fname in os.listdir(tree_dir):
        m = nwck_pattern.match(fname)
        if m:
            nwck_ids[m.group(1)] = os.path.join(tree_dir, fname)

    common_ids = sorted(set(fasta_ids) & set(nwck_ids))

    dropped_fasta = set(fasta_ids) - set(nwck_ids)
    dropped_nwck = set(nwck_ids) - set(fasta_ids)
    if dropped_fasta:
        logger.warning("Dropped %d FASTA files with no matching tree: %s",
                        len(dropped_fasta), dropped_fasta)
    if dropped_nwck:
        logger.warning("Dropped %d Newick files with no matching alignment: %s",
                        len(dropped_nwck), dropped_nwck)

    pairs = [(fasta_ids[id_], nwck_ids[id_]) for id_ in common_ids]
    logger.info("Discovered %d matched alignment/tree pairs", len(pairs))
    return pairs


# fasta parser

def parse_fasta(filepath):
    """
    parses a fasta file into an ordered list of (name, sequence) tuples
    lines starting with '>' are headers, subsequent lines are concatenated as the sequence
    entries with no sequence are skipped
    """
    entries = []
    current_name = None
    current_seq = []

    with open(filepath) as f:
        for line in f:
            line = line.rstrip("\n\r")
            if line.startswith(">"):
                # flush previous entry on each new header
                if current_name is not None:
                    seq = "".join(current_seq)
                    if seq:
                        entries.append((current_name, seq))
                current_name = line[1:].strip()
                current_seq = []
            else:
                current_seq.append(line.strip())

    # flush the final entry
    if current_name is not None:
        seq = "".join(current_seq)
        if seq:
            entries.append((current_name, seq))

    return entries


# newick pairwise distances

def compute_pairwise_distances(nwck_path, leaf_order):
    """
    parses a newick tree and computes patristic (branch-length) distances for every pair of leaves,
    ordered to match leaf_order so the result is consistent with pair_matrix in the model
    arguments:
        nwck_path:  path to a newick (.nwck) file
        leaf_order: list of taxon names in the same order they appear in the fasta file
    returns:
        FloatTensor of shape (num_pairs,)
    """
    tree = dendropy.Tree.get(path=nwck_path, schema="newick")
    pdm = tree.phylogenetic_distance_matrix()

    # map taxon label -> Taxon object so we can index into the distance matrix by name
    taxon_map = {taxon.label: taxon for taxon in tree.taxon_namespace}

    distances = []
    for i, j in combinations(range(len(leaf_order)), 2):
        t_i = taxon_map[leaf_order[i]]
        t_j = taxon_map[leaf_order[j]]
        distances.append(pdm(t_i, t_j))

    return np.array(distances, dtype=np.float32)


# preprocessing worker (module-level so it is picklable by ProcessPoolExecutor)

def _preprocess_one(item):
    """parse one fasta/newick pair and compute pairwise distances — runs in a worker process"""
    fasta_path, nwck_path = item
    entries = parse_fasta(fasta_path)
    names = [name for name, _ in entries]
    sequences = [seq for _, seq in entries]
    distances = compute_pairwise_distances(nwck_path, names)
    return sequences, distances


# dataset

class PhyloDataset(Dataset):
    """
    dataset that pairs fasta alignments with newick-derived pairwise distance targets.
    all parsing, distance computation, and tokenization are performed eagerly at
    construction time using a multiprocessing pool so the GPU does not idle at runtime.
    arguments:
        alignment_dir:          directory containing {id}_50_tips.fasta files
        tree_dir:               directory containing {id}_50_tips.nwk files
        tokenizer:              a Tokenizer instance used to encode sequences
        num_preprocess_workers: worker processes for parallel preprocessing
                                (defaults to os.cpu_count())
    each sample returns:
        alignment : LongTensor of shape (R, C)       tokenised alignment
        distances : FloatTensor of shape (num_pairs,) patristic distances
    """

    def __init__(self, alignment_dir, tree_dir, tokenizer: Tokenizer, num_preprocess_workers = 0):
        pairs = discover_pairs(alignment_dir, tree_dir)
        n_workers = num_preprocess_workers if num_preprocess_workers > 0 else len(os.sched_getaffinity(0))

        logger.info("Preprocessing %d samples with %d worker processes...", len(pairs), n_workers)
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            raw = pool.map(_preprocess_one, pairs, chunksize=max(1, len(pairs) // (n_workers * 4)))

        logger.info("Tokenizing %d samples...", len(raw))
        self.data = [
            (tokenizer.encode(sequences), torch.from_numpy(distances))
            for sequences, distances in raw
        ]
        logger.info("Dataset ready — %d samples loaded into memory", len(self.data))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.data[idx]


# dataloader factory

def create_dataloaders(
    train_alignment_dir,
    train_tree_dir,
    val_alignment_dir,
    val_tree_dir,
    tokenizer: Tokenizer,
    batch_size = 32,
    num_workers = 0,
    num_preprocess_workers = 0,
):
    """
    builds train and val dataloaders from the four data directories
    arguments:
        train_alignment_dir:    path to train/alignment
        train_tree_dir:         path to train/trees
        val_alignment_dir:      path to val/alignment
        val_tree_dir:           path to val/trees
        tokenizer:              a Tokenizer instance
        batch_size:             samples per batch
        num_workers:            dataloader worker processes
        num_preprocess_workers: worker processes for parallel preprocessing
                                (defaults to os.cpu_count())
    returns:
        (train_loader, val_loader)
    """
    train_ds = PhyloDataset(train_alignment_dir, train_tree_dir, tokenizer,
                            num_preprocess_workers=num_preprocess_workers)
    val_ds = PhyloDataset(val_alignment_dir, val_tree_dir, tokenizer,
                          num_preprocess_workers=num_preprocess_workers)

    persistent = num_workers > 0
    # val loader uses fewer workers — validation only needs ~30 samples at a time
    val_workers = min(num_workers, 2)
    val_persistent = val_workers > 0
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                              persistent_workers=persistent, prefetch_factor=1 if persistent else None)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=val_workers,
                            persistent_workers=val_persistent, prefetch_factor=1 if val_persistent else None)

    return train_loader, val_loader
