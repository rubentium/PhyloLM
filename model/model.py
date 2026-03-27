import torch
import torch.nn as nn
import torch.nn.functional as F
from .axial_transfomer import Axial_Transformer

class PhyloLM(nn.Module):
    """
    the main model class for the phylogenetic language model
    the model consists of multiple transformer blocks, each containing a multi-head attention module and a feedforward network
    arguments:
        num_blocks: number of transformer blocks in the model
        h_dim: hidden dimension of the transformer blocks
        num_heads: number of attention heads in each block
        dropout: dropout rate for attention probabilities and feedforward layers
    """
    def __init__(self, num_rows, num_cols, num_blocks, h_dim, num_heads, dropout=0.1):
        super(PhyloLM, self).__init__()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.embedding = nn.Embedding(21, h_dim)  # 20 amino acids + gap token
        self.pair_matrix = pair_matrix(num_rows, device=device)
        self.blocks = nn.ModuleList([
            Axial_Transformer(h_dim, num_heads, dropout) for _ in range(num_blocks)
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
            x = block(x, mask)
        x = self.penultimate_ffn(x).squeeze(-1)  # (B, num_pairs, C)
        x = self.ultimate_ffn(x).squeeze(-1)  # (B, num_pairs)
        return x
    
def pair_matrix(rows, device):
    """
    generates a pairwise mask for the input sequences to be used in the attention mechanism
    """
    pairs = torch.combinations(torch.arange(rows), r=2)
    num_pairs = pairs.size(0)
    mask = torch.zeros((num_pairs, rows))
    mask.scatter_(1, pairs, 1.0)
    mask = mask.to(device)
    return mask
