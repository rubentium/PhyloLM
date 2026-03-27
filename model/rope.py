import torch
import torch.nn as nn

class RotaryEmbedding(nn.Module):
    """
    rotary position embedding (RoPE) module
    encodes position information by rotating query and key vectors in the complex plane,
    allowing attention scores to be a function of relative position rather than absolute position
    arguments:
        dim: the head dimension (must be even)
        base: the base for the geometric sequence of frequencies (default 10000)
    """
    def __init__(self, dim, base=10000):
        super().__init__()
        assert dim % 2 == 0, "RoPE head dimension must be even"
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, seq_len, device):
        """
        compute cos and sin rotation matrices for a given sequence length
        returns:
            cos, sin: tensors of shape (seq_len, dim)
        """
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)       # (seq_len, dim/2)
        emb = torch.cat([freqs, freqs], dim=-1)     # (seq_len, dim)
        return emb.cos(), emb.sin()


def rotate_half(x):
    """split the last dimension in half and rotate: [x1, x2] -> [-x2, x1]"""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_emb(q, k, cos, sin):
    """
    apply rotary embeddings to query and key tensors
    arguments:
        q, k: tensors of shape (..., seq_len, head_dim)
        cos, sin: tensors of shape (seq_len, head_dim), from RotaryEmbedding.forward
    returns:
        q_rot, k_rot: rotated tensors with the same shape as q and k
    """
    # unsqueeze to broadcast over all leading batch/head dims
    cos = cos.unsqueeze(0)  # (1, seq_len, head_dim)
    sin = sin.unsqueeze(0)  # (1, seq_len, head_dim)
    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)
    return q_rot, k_rot
