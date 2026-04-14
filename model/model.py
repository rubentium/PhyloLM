import torch
import torch.nn as nn
from .axial_transfomer import Axial_Transformer
import torch.nn.functional as F

class PhyloLM(nn.Module):
    """
    the main model class for the phylogenetic language model
    the model consists of multiple transformer blocks, each containing a multi-head attention module and a feedforward network
    arguments:
        num_blocks: number of transformer blocks in the model
        h_dim: hidden dimension of the transformer blocks
        num_heads: number of attention heads in each block
        vocab_size: size of the vocabulary for the embedding layer
        dropout: dropout rate for attention probabilities and feedforward layers
        type: type of attention mechanism to use ("sparse" or "dense")
    """
    def __init__(self, num_rows, num_cols, num_blocks, h_dim, num_heads, vocab_size, dropout=0.1, att_type="sparse", num_random_blocks=1):
        super(PhyloLM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, h_dim)  # esm2 vocab size
        self.num_blocks = num_blocks
        self.att_type = att_type
        pair_matrix, padding = pair_matrix_f(num_rows, type=att_type)
        self.register_buffer('pair_matrix', pair_matrix)
        self.register_buffer('pair_padding', torch.tensor(padding))
        self.num_pairs = self.pair_matrix.size(0)
        self.blocks = nn.ModuleList([
            Axial_Transformer(h_dim, 
                              num_heads, 
                              self.num_pairs, 
                              num_cols, 
                              dropout, 
                              att_type=att_type, 
                              padding=self.pair_padding, 
                              num_random_blocks=num_random_blocks) 
            for _ in range(num_blocks)
        ])

        # dont ask me why i gave these two ffns these weird names, i just felt like it ¯\_(ツ)_/¯
        self.penultimate_ffn = nn.Sequential(
            nn.Linear(h_dim, h_dim * 4),
            nn.GELU(),
            nn.Linear(h_dim * 4, 1)
        )
        self.ultimate_ffn = nn.Linear(num_cols, 1)

    def forward(self, x, sparse_indices=None, mask=None, random_perm=None):
        # input is (B, R, C)
        x = self.embedding(x)      # (B, R, C, H)
        x = x.permute(0, 3, 1, 2)  # (B, H, R, C)
        x = self.pair_matrix @ x   # (B, H, num_pairs, C)

        if random_perm is not None:
            x = x.index_select(2, random_perm)

        x = x.permute(0, 2, 3, 1)                # (B, num_pairs, C, H)
        for i, block in enumerate(self.blocks):
            x = block(x, idx=sparse_indices[i] if sparse_indices is not None else None, mask=mask)
            # x = torch.utils.checkpoint.checkpoint(block, x, sparse_indices[i] if sparse_indices is not None else None, mask, use_reentrant=False)
        x = self.penultimate_ffn(x).squeeze(-1)  # (B, num_pairs, C)
        x = self.ultimate_ffn(x).squeeze(-1)     # (B, num_pairs)
        return x
    
def pair_matrix_f(rows, type="sparse"):
    """
    generates a pairwise mask for the input sequences to be used in the attention mechanism
    """
    pairs = torch.combinations(torch.arange(rows), r=2)
    num_pairs = pairs.size(0)
    mask = torch.zeros((num_pairs, rows))
    mask.scatter_(1, pairs, 1.0)
    padding = (0, 0, 27, 28) if type == "sparse" else (0, 0, 0, 0)  # padding to 1228 for tiling in flex attention
    mask = F.pad(mask, padding, value=0)  # pading to 1228 for tiling in flex attention
    return mask.to(dtype=torch.bfloat16), padding