# import torch for tensors
import torch

# use pathlib to work with folders and file paths
from pathlib import Path

# import dendropy for tree parsing
import dendropy

# import combinations to generate all sequence pairs
from itertools import combinations

# define amino acid alphabet
# includes 20 standard amino acis, x for unknown, - for gap
alphabet = "ARNDCQEGHILKMFPSTWYVX-"

# create mapping from character to index
char_to_index = {}
for i, char in enumerate(alphabet):
    char_to_index[char] = i


def read_fasta(filepath):

    """
    Read a FASTA alignment file.

    Input:
        filepath: path to one MSA file

    Output:
        ids: list of sequence names
        sequences: list of aligned sequences
    """

    # store sequence names
    ids = []

    # store sequence strings
    sequences = []

    # temporary sequence builder
    current_seq = ""

    # open the FASTA file in text mode
    with open(filepath, "r") as f:

        # read file line by line
        for line in f:
            line = line.strip()

            # skip empty lines
            if line == "":
                continue

            # header line starts with ">"
            if line.startswith(">"):

                # if we have a current sequence, save it
                if current_seq != "":
                    sequences.append(current_seq)
                    current_seq = ""

                # save the sequence name (without ">")
                ids.append(line[1:])

            else:
                # add line to current sequence
                current_seq += line

        # save the last sequence after file ends
        if current_seq != "":
            sequences.append(current_seq)

    # return ids and aligned sequences
    return ids, sequences

def msa_to_indices(sequences):

    """ 
    Convert a list of aligned sequences into integer indices.
    
    Input:
        sequences: list of strings
        
    Output:
        tensor of shape (S, L)
        S = number of sequences
        L = length of each sequence
    """

    # check that we actually have sequences
    if len(sequences) == 0:
        raise ValueError("No sequences provided")
    
    # number of sequences
    S = len(sequences)

    # alignmet length from first sequences
    L = len(sequences[0])

    # make sure all sequences have the same length
    for seq in sequences:
        if len(seq) != L:
            raise ValueError("All sequences in the MSA must have the same length")

    # create integer tensor
    indices = torch.zeros((S, L), dtype=torch.long)

    # fill tensor entry-by-entry
    for i in range(S):
        for j in range(L):

            # get one character
            char = sequences[i][j]

            # replace unknown symbols with X
            if char not in char_to_index:
                char = "X"

            # store integer index
            indices[i, j] = char_to_index[char]

    return indices

def one_hot_encode(indices):

    """
    Convert integer indices to one-hot encoding.

    Input:
        indices: tensor of shape (S, L) 

    Output:
        one_hot: tensor of shape (S, L, A) 
        A = size of amino acid alphabet
    """

    # size of amino acid alphabet
    A = len(alphabet)

    # use PyTorch one-hot utility
    x = torch.nn.functional.one_hot(indices, num_classes=A)

    # convert from integer type to float
    x = x.float()

    return x

def load_alignment(filepath):

    """
    Full MSA preprocessing pipeline.
    
    Input:
        filepath: path to one FASTA alignment file
        
    Output:
        x: one-hot tensor of shape (S, L, A)
        idx: list of sequence names
    """

    # read FASTA file
    ids, sequences = read_fasta(filepath)

    # convert sequence characters to integer indices
    indices = msa_to_indices(sequences)

    # convert integer indices to one-hot tensor
    x = one_hot_encode(indices)

    return x, ids

def load_distance_matrix(filepath, ids):

    """
    Load a tree file and compute pairwise distances between sequences.
    
    Input:
        filepath: path to tree file (Newick format)
        ids: list of sequence names (must match MSA order)
        
    Output:
        y: tensor of shape (P, )
        P = number of sequence pairs = S choose 2
    """

    # store distances 
    distances = []

    # read tree file
    with open(filepath, "r") as f:
        tree = dendropy.Tree.get(file=f, schema="newick")
    
    # get taxa (sequence labels)
    taxa = tree.taxon_namespace

    # get distance matrix object
    dm = tree.phylogenetic_distance_matrix()

    # loop over all unique pairs (i<j)
    for id1, id2 in combinations(ids, 2):

        # get taxon objects for the two sequences
        taxon1 = taxa.get_taxon(label=id1)
        taxon2 = taxa.get_taxon(label=id2)

        # compute distance between the two taxa
        dist = dm(taxon1, taxon2)

        # store distance
        distances.append(dist)

    # convert distances to tensor
    y = torch.tensor(distances, dtype=torch.float)

    return y