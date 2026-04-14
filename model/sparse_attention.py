import torch
import torch.nn as nn
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

_MASK_POOL_SIZE = 1000

class SparseAttention(nn.Module):
    """
    Single-pass sparse attention: each query block attends to its own diagonal
    block (local) plus num_random_blocks randomly chosen other blocks (global),
    unified in a single flex_attention call.

    A pool of _MASK_POOL_SIZE block masks is built at init time, each with a
    different random assignment of global blocks per query block. During training
    one mask is sampled per forward call; during eval mask 0 is always used,
    giving deterministic behaviour.

    interface mirrors `Attention` in axial_transfomer.py:
        forward(x: [B, extra, S, h_dim], mask=None) -> [B, extra, S, h_dim]

    args:
        h_dim:             Hidden dimension (must be divisible by num_heads).
        num_heads:         Number of attention heads.
        seq_len:           Sequence length to attend over (must be divisible by block_size).
        block_size:        Size of each local block tile (default 128).
        dropout:           Dropout probability applied to the output projection.
        padding:           (pad_top, pad_bottom, pad_left, pad_right) token counts to exclude.
        num_random_blocks: Number of extra random (off-diagonal) blocks each query block
                           attends to. Clipped to [0, num_blocks-1]. Default 1.
    """

    def __init__(
        self,
        h_dim: int,
        num_heads: int,
        seq_len: int,
        block_size: int = 128,
        dropout: float = 0.1,
        padding=(0, 0, 0, 0),
        num_random_blocks: int = 1,
    ):
        super().__init__()
        assert h_dim % num_heads == 0, "h_dim must be divisible by num_heads"
        assert seq_len % block_size == 0, (
            f"seq_len ({seq_len}) must be divisible by block_size ({block_size})"
        )
        assert seq_len // block_size >= 2, (
            "seq_len must span at least 2 blocks for sparse attention to be meaningful"
        )

        self.num_heads = num_heads
        self.head_dim = h_dim // num_heads
        self.seq_len = seq_len
        self.block_size = block_size
        self.num_blocks = seq_len // block_size
        self.padding = padding
        self.num_random_blocks = max(0, min(num_random_blocks, self.num_blocks - 1))

        self.query = nn.Linear(h_dim, h_dim)
        self.key = nn.Linear(h_dim, h_dim)
        self.value = nn.Linear(h_dim, h_dim)
        self.out = nn.Linear(h_dim, h_dim)
        self.drop = nn.Dropout(dropout)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if not hasattr(self, '_rt_pool'):
            self.register_buffer("_rt_pool", torch.zeros(
                (_MASK_POOL_SIZE, self.num_blocks, self.num_random_blocks),
                dtype=torch.int8, device=device,
            ))
            self._mask_pool = self._build_mask_pool(device)

    def _build_mask_pool(self, device: torch.device) -> list:
        """Build _MASK_POOL_SIZE masks, each combining the diagonal block with
        num_random_blocks randomly chosen other blocks per query block.

        Random block selection uses randperm+slice+shift: O(num_blocks) per query
        block, always produces exactly num_random_blocks unique non-diagonal indices.
        """
        pad_start = int(self.padding[2])
        pad_end = int(self.seq_len - self.padding[3])
        block_size = self.block_size
        num_blocks = self.num_blocks
        num_random_blocks = self.num_random_blocks
        pool = []

        def make_mask_mod(rt, bs, ps, pe, nrb):
            """Factory to avoid closure-capture issues in the build loop."""
            def mask_mod(b, h, q_idx, kv_idx):
                not_padding = (q_idx >= ps) & (q_idx < pe) & \
                              (kv_idx >= ps) & (kv_idx < pe)
                q_block = q_idx // bs
                kv_block = kv_idx // bs
                is_local = q_block == kv_block
                if nrb == 0:
                    return not_padding & is_local
                # unrolled OR over random blocks (nrb is a compile-time constant)
                is_global = kv_block == rt[q_block, 0]
                for j in range(1, nrb):
                    is_global = is_global | (kv_block == rt[q_block, j])
                return not_padding & (is_local | is_global)
            return mask_mod

        for i in range(_MASK_POOL_SIZE):
            if num_random_blocks > 0:
                for qb in range(num_blocks):
                    # randperm over num_blocks-1 candidates, then shift values
                    # >= qb by +1 to skip the diagonal block
                    perm = torch.randperm(num_blocks - 1)[:num_random_blocks].clone()
                    perm[perm >= qb] += 1
                    self._rt_pool[i, qb] = perm.to(dtype=torch.int8)

            pool.append(create_block_mask(
                make_mask_mod(self._rt_pool[i], block_size, pad_start, pad_end, num_random_blocks),
                B=None, H=None,
                Q_LEN=self.seq_len, KV_LEN=self.seq_len, device=device,
            ))
        return pool

    def forward(self, x: torch.Tensor, idx=None, mask=None) -> torch.Tensor:
        batch_size, extra, seq_len, h_dim = x.size()

        q = self.query(x).view(batch_size * extra, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(batch_size * extra, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(batch_size * extra, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        if idx is None or (not self.training):
            idx = 0
        block_mask = self._mask_pool[idx]
        # print(block_mask)
        out = flex_attention(q, k, v, block_mask=block_mask)
        out = out.transpose(1, 2).contiguous().view(batch_size, extra, seq_len, h_dim)
        return self.drop(self.out(out))
