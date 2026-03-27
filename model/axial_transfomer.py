import torch
import torch.nn as nn
import torch.nn.functional as F
from model.rope import RotaryEmbedding, apply_rotary_emb

class Attention(nn.Module):
    """
    the multi-head attention module for the encoder transformer block
    the module supports masked language modeling (mlm) by optionally masking tokens
    arguments:
        h_dim: hidden dimension of the transformer block
        num_heads: number of attention heads
        dropout: dropout rate for attention probabilities
        use_rope: whether to apply rotary position embeddings to queries and keys
    """
    def __init__(self, h_dim, num_heads, dropout=0.1, use_rope=True):
        super(Attention, self).__init__()
        assert h_dim % num_heads == 0, "Hidden dimension must be divisible by number of heads"
        self.num_heads = num_heads
        self.head_dim = h_dim // num_heads

        self.query = nn.Linear(h_dim, h_dim)
        self.key = nn.Linear(h_dim, h_dim)
        self.value = nn.Linear(h_dim, h_dim)
        self.out = nn.Linear(h_dim, h_dim)
        self.dropout = nn.Dropout(dropout)
        self.rope = RotaryEmbedding(self.head_dim) if use_rope else None

    def forward(self, x, mask=None):
        batch_size, rows, cols, h_dim = x.size()

        q = self.query(x).view(batch_size, rows, cols, self.num_heads, self.head_dim).transpose(2, 3)  # (B, R, H, C, D)
        k = self.key(x).view(batch_size, rows, cols, self.num_heads, self.head_dim).transpose(2, 3)    # (B, R, H, C, D)
        v = self.value(x).view(batch_size, rows, cols, self.num_heads, self.head_dim).transpose(2, 3)  # (B, R, H, C, D)

        if self.rope is not None:
            cos, sin = self.rope(cols, device=x.device)  # (C, D)
            q, k = apply_rotary_emb(q, k, cos, sin)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (B, R, H, C, C)
        
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        out = torch.matmul(attn_probs, v)  # (B, H, R, C, D)
        out = out.transpose(2, 3).contiguous().view(batch_size, rows, cols, h_dim)
        out = self.out(out)
        return out


class Axial_Transformer(nn.Module):
    """
    the axial transformer module for the encoder transformer block
    the module applies attention along rows and columns separately to capture long-range dependencies in both dimensions
    this is implemented in accordance with the Pre-LM Transfomrer architecture
    arguments:
        h_dim: hidden dimension of the transformer block
        num_heads: number of attention heads
        dropout: dropout rate for attention probabilities
        use_rope: whether to apply rotary position embeddings (default True)
    """
    def __init__(self, h_dim, num_heads, dropout=0.1, use_rope=True):
        super(Axial_Transformer, self).__init__()
        self.row_attention = Attention(h_dim, num_heads, dropout, use_rope)
        self.col_attention = Attention(h_dim, num_heads, dropout, use_rope)
        
        self.row_norm = nn.LayerNorm(h_dim)
        self.col_norm = nn.LayerNorm(h_dim)
        self.ffn_norm = nn.LayerNorm(h_dim)

        self.row_ff = nn.Sequential(
            nn.Linear(h_dim, h_dim * 4),
            nn.GELU(),
            nn.Linear(h_dim * 4, h_dim)
        )

    def forward(self, x, mask=None):
        # apply row-wise attention
        row_x = self.row_norm(x)
        row_attn_out = self.row_attention(row_x, mask)
        x = x + row_attn_out

        # apply column-wise attention
        col_x = self.col_norm(x)
        col_attn_out = self.col_attention(col_x.transpose(1, 2), mask)
        x = x + col_attn_out.transpose(1, 2)
        
        # apply feedforward network
        ffn_x = self.ffn_norm(x)
        ffn_out = self.row_ff(ffn_x)
        x = x + ffn_out
        return x