import torch
import torch.nn as nn

def rotate_half(x):
    """split the last dimension in half and rotate: [x1, x2] -> [-x2, x1]"""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)

class RotaryEmbedding(nn.Module):
    """
    rotary position embedding (RoPE) module
    encodes position information by rotating query and key vectors in the complex plane,
    allowing attention scores to be a function of relative position rather than absolute position
    arguments:
        dim: the head dimension (must be even)
        base: the base for the geometric sequence of frequencies (default 10000)
    """
    def __init__(self, dim, seq_len, device, base=10000):
        super().__init__()
        assert dim % 2 == 0, "RoPE head dimension must be even"
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim)).to(device)
        self.register_buffer("inv_freq", inv_freq)
        
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)       # (seq_len, dim/2)
        emb = torch.cat([freqs, freqs], dim=-1)     # (seq_len, dim)
        self.register_buffer("cos_emb", emb.cos())
        self.register_buffer("sin_emb", emb.sin())

    def forward(self, q, k):
        """
        apply rotary embeddings to query and key tensors
        arguments:
            q, k: tensors of shape (..., seq_len, head_dim)
            cos, sin: tensors of shape (seq_len, head_dim), from RotaryEmbedding.forward
        returns:
            q_rot, k_rot: rotated tensors with the same shape as q and k
        """
        # unsqueeze to broadcast over all leading batch/head dims
        cos = self.cos_emb.unsqueeze(0)  # (1, seq_len, head_dim)
        sin = self.sin_emb.unsqueeze(0)  # (1, seq_len, head_dim)
        q_rot = (q * cos) + (rotate_half(q) * sin)
        k_rot = (k * cos) + (rotate_half(k) * sin)
        return q_rot, k_rot