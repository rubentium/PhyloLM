import torch
import torch.nn as nn
import torch.nn.functional as F
from .rope import RotaryEmbedding
from .sparse_attention import SparseAttention

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
    def __init__(self, h_dim, num_heads, seq_len, dropout=0.1, use_rope=True):
        super(Attention, self).__init__()
        assert h_dim % num_heads == 0, "Hidden dimension must be divisible by number of heads"
        self.num_heads = num_heads
        self.head_dim = h_dim // num_heads

        self.query = nn.Linear(h_dim, h_dim)
        self.key = nn.Linear(h_dim, h_dim)
        self.value = nn.Linear(h_dim, h_dim)
        self.out = nn.Linear(h_dim, h_dim)
        self.dropout = nn.Dropout(dropout)
        self.rope = RotaryEmbedding(self.head_dim, seq_len=seq_len, device='cuda' if torch.cuda.is_available() else 'cpu') if use_rope else None

    def forward(self, x, mask=None, idx=None):
        """
        this is the flash attention implementation of attanetion due to
        axial attention it's near impossible to scale the model to anything beyond 
        a few hundred thousand parameters even on 80GB VRAM H100s
        
        Note: Do not remove idx, its not used here, but it's there because sparse attention
        version of this class needs it, and its easier to just have it as an optional argument
        than to make a whole new class for dense row and column attentions without idx
        """
        batch_size, extra, seq_len, h_dim = x.size()

        q = self.query(x).view(batch_size, extra, seq_len, self.num_heads, self.head_dim).transpose(2, 3)  # (B, E, H, S, D)
        k = self.key(x).view(batch_size, extra, seq_len, self.num_heads, self.head_dim).transpose(2, 3)    # (B, E, H, S, D)
        v = self.value(x).view(batch_size, extra, seq_len, self.num_heads, self.head_dim).transpose(2, 3)  # (B, E, H, S, D)

        if self.rope is not None:
            q, k = self.rope(q, k)  # (C, D)

        attn_mask = mask.bool() if mask is not None else None
        # SDPA only dispatches Flash/mem-efficient attention for 4D tensors (B, H, S, D).
        # the tensors are 5D (B, extra, H, S, D), so we flatten the leading two dims first.
        q = q.flatten(0, 1)  # (B*rows, H, S, D)
        k = k.flatten(0, 1)
        v = v.flatten(0, 1)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=self.dropout.p if self.training else 0.0)
        out = out.view(batch_size, extra, self.num_heads, seq_len, self.head_dim)
        out = out.transpose(2, 3).contiguous().view(batch_size, extra, seq_len, h_dim)
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
    def __init__(self, h_dim, num_heads, rows, cols, dropout=0.1, use_rope=True, att_type="sparse", padding=(0, 0, 0, 0), num_random_blocks=1):
        super(Axial_Transformer, self).__init__()
        if att_type == "sparse":
            self.row_attention = SparseAttention(h_dim, num_heads, rows, dropout=dropout, padding=padding, num_random_blocks=num_random_blocks) # this one is not a dual class (only operates over row) so no rope passed
        else:
            self.row_attention = Attention(h_dim, num_heads, rows, dropout, use_rope=False)  # no rope for row attention since it operates on pairs, not sequences

        self.col_attention = Attention(h_dim, num_heads, cols, dropout, use_rope)
        self.row_norm = nn.RMSNorm(h_dim)
        self.col_norm = nn.RMSNorm(h_dim)
        self.ffn_norm = nn.RMSNorm(h_dim)

        self.row_ff = nn.Sequential(
            nn.Linear(h_dim, h_dim * 4),
            nn.GELU(),
            nn.Linear(h_dim * 4, h_dim)
        )

    def forward(self, x, idx=None, mask=None):
        # apply row-wise attention
        row_x = self.row_norm(x)
        row_attn_out = self.row_attention(row_x.transpose(1, 2), idx=idx, mask=mask)
        x = x + row_attn_out.transpose(1, 2)

        # apply column-wise attention
        col_x = self.col_norm(x)
        col_attn_out = self.col_attention(col_x, mask=mask)
        x = x + col_attn_out
        
        # apply feedforward network
        ffn_x = self.ffn_norm(x)
        ffn_out = self.row_ff(ffn_x)
        x = x + ffn_out
        return x