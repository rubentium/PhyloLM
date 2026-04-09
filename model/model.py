import torch
import torch.nn as nn
from .axial_transfomer import Axial_Transformer
import torch.utils.checkpoint

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
    """
    def __init__(self, num_rows, num_cols, num_blocks, h_dim, num_heads, vocab_size, dropout=0.1):
        super(PhyloLM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, h_dim)  # esm2 vocab size
        self.register_buffer('pair_matrix', pair_matrix(num_rows))
        num_pairs = self.pair_matrix.size(0)
        self.blocks = nn.ModuleList([
            Axial_Transformer(h_dim, num_heads, num_pairs, num_cols, dropout) for _ in range(num_blocks)
        ])
        
        # dont ask me why i gave these two ffns these weird names, i just felt like it ¯\_(ツ)_/¯
        self.penultimate_ffn = nn.Sequential(
            nn.Linear(h_dim, h_dim * 4),
            nn.GELU(),
            nn.Linear(h_dim * 4, 1)
        )
        self.ultimate_ffn = nn.Linear(num_cols, 1)

    def forward(self, x, mask=None):
        # input is (B, R, C)
        x = self.embedding(x)  # (B, R, C, H)
        x = x.permute(0, 3, 1, 2)  # (B, H, R, C)
        x = self.pair_matrix @ x   # (B, H, num_pairs, C)
        x = x.permute(0, 2, 3, 1)  # (B, num_pairs, C, H)
        for block in self.blocks:
            x = torch.utils.checkpoint.checkpoint(block, x, mask, use_reentrant=False)
        x = self.penultimate_ffn(x).squeeze(-1)  # (B, num_pairs, C)
        x = self.ultimate_ffn(x).squeeze(-1)  # (B, num_pairs)
        return x
    
def pair_matrix(rows):
    """
    generates a pairwise mask for the input sequences to be used in the attention mechanism
    """
    pairs = torch.combinations(torch.arange(rows), r=2)
    num_pairs = pairs.size(0)
    mask = torch.zeros((num_pairs, rows))
    mask.scatter_(1, pairs, 1.0)
    return mask.to(dtype=torch.bfloat16)
