import os
import re
import logging
from itertools import combinations
from turtle import st

import torch
from torch.utils.data import Dataset, DataLoader
import dendropy

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

    def encode(self, sequences: list[str]) -> torch.Tensor:
        raise NotImplementedError("Tokenizer.encode() must be implemented")


# file discovery

def discover_pairs(alignment_dir, tree_dir):
    """
    finds matching fasta/newick file pairs by extracting {id} from filenames
    of the form {id}_50.fasta and {id}_50.nwck — any unpaired files are dropped with a warning
    returns a sorted list of (fasta_path, nwck_path) tuples
    """
    fasta_pattern = re.compile(r"^(.+)_50\.fasta$")
    nwck_pattern = re.compile(r"^(.+)_50\.nwck$")

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

    return torch.tensor(distances, dtype=torch.float32)


# dataset

class PhyloDataset(Dataset):
    """
    dataset that pairs fasta alignments with newick-derived pairwise distance targets
    arguments:
        alignment_dir: directory containing {id}_50.fasta files
        tree_dir:      directory containing {id}_50.nwck files
        tokenizer:     a Tokenizer instance used to encode sequences
    each sample returns:
        alignment : LongTensor of shape (R, C)     tokenised alignment
        distances : FloatTensor of shape (num_pairs,) patristic distances
    """

    def __init__(self, alignment_dir, tree_dir, tokenizer: Tokenizer):
        self.pairs = discover_pairs(alignment_dir, tree_dir)
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        fasta_path, nwck_path = self.pairs[idx]

        entries = parse_fasta(fasta_path)
        names = [name for name, _ in entries]
        sequences = [seq for _, seq in entries]

        alignment = self.tokenizer.encode(sequences)          # (R, C)
        distances = compute_pairwise_distances(nwck_path, names)  # (num_pairs,)

        return alignment, distances


# dataloader factory

def create_dataloaders(
    train_alignment_dir,
    train_tree_dir,
    val_alignment_dir,
    val_tree_dir,
    tokenizer: Tokenizer,
    batch_size = 32,
    num_workers = 0,
):
    """
    builds train and val dataloaders from the four data directories
    arguments:
        train_alignment_dir: path to train/alignment
        train_tree_dir:      path to train/trees
        val_alignment_dir:   path to val/alignment
        val_tree_dir:        path to val/trees
        tokenizer:           a Tokenizer instance
        batch_size:          samples per batch
        num_workers:         dataloader worker processes
    returns:
        (train_loader, val_loader)
    """
    train_ds = PhyloDataset(train_alignment_dir, train_tree_dir, tokenizer)
    val_ds = PhyloDataset(val_alignment_dir, val_tree_dir, tokenizer)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader
