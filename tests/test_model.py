import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')))

import torch
import unittest
from model.axial_transfomer import Attention, Axial_Transformer
from model.model import PhyloLM, pair_matrix
from model.rope import RotaryEmbedding, rotate_half, apply_rotary_emb

B, R, C, H, HEADS = 2, 4, 6, 32, 4
NUM_PAIRS = R * (R - 1) // 2  # 6


# ---------------------------------------------------------------------------
# RoPE
# ---------------------------------------------------------------------------

class TestRotaryEmbedding(unittest.TestCase):

    def setUp(self):
        self.dim = 16
        self.seq_len = 10
        self.rope = RotaryEmbedding(self.dim)

    def test_output_shapes(self):
        cos, sin = self.rope(self.seq_len, device=torch.device('cpu'))
        self.assertEqual(cos.shape, (self.seq_len, self.dim))
        self.assertEqual(sin.shape, (self.seq_len, self.dim))

    def test_no_nan(self):
        cos, sin = self.rope(self.seq_len, device=torch.device('cpu'))
        self.assertFalse(torch.isnan(cos).any())
        self.assertFalse(torch.isnan(sin).any())

    def test_values_in_range(self):
        cos, sin = self.rope(self.seq_len, device=torch.device('cpu'))
        self.assertTrue((cos >= -1.0).all() and (cos <= 1.0).all())
        self.assertTrue((sin >= -1.0).all() and (sin <= 1.0).all())

    def test_pythagorean_identity(self):
        # cos²(θ) + sin²(θ) == 1 for every element
        cos, sin = self.rope(self.seq_len, device=torch.device('cpu'))
        identity = cos**2 + sin**2
        self.assertTrue(torch.allclose(identity, torch.ones_like(identity), atol=1e-6))

    def test_position_zero_is_identity(self):
        # at position 0 all angles are 0, so cos == 1 and sin == 0
        cos, sin = self.rope(self.seq_len, device=torch.device('cpu'))
        self.assertTrue(torch.allclose(cos[0], torch.ones(self.dim), atol=1e-6))
        self.assertTrue(torch.allclose(sin[0], torch.zeros(self.dim), atol=1e-6))

    def test_different_seq_lengths(self):
        for length in [1, 8, 64]:
            cos, sin = self.rope(length, device=torch.device('cpu'))
            self.assertEqual(cos.shape[0], length)


class TestRotateHalf(unittest.TestCase):

    def test_output_shape_preserved(self):
        x = torch.randn(2, 4, 8, 16)
        self.assertEqual(rotate_half(x).shape, x.shape)

    def test_double_rotation_negates(self):
        # rotate_half(rotate_half(x)) == -x
        x = torch.randn(3, 10, 16)
        self.assertTrue(torch.allclose(rotate_half(rotate_half(x)), -x, atol=1e-6))

    def test_correct_permutation(self):
        # for x = [x1 | x2], rotate_half should give [-x2 | x1]
        x1 = torch.randn(4, 8)
        x2 = torch.randn(4, 8)
        x = torch.cat([x1, x2], dim=-1)
        expected = torch.cat([-x2, x1], dim=-1)
        self.assertTrue(torch.allclose(rotate_half(x), expected))


class TestApplyRotaryEmb(unittest.TestCase):

    def setUp(self):
        self.dim = 16
        self.seq_len = 10
        self.rope = RotaryEmbedding(self.dim)
        self.cos, self.sin = self.rope(self.seq_len, device=torch.device('cpu'))

    def test_output_shapes_preserved(self):
        q = torch.randn(2, 4, self.seq_len, self.dim)
        k = torch.randn(2, 4, self.seq_len, self.dim)
        q_rot, k_rot = apply_rotary_emb(q, k, self.cos, self.sin)
        self.assertEqual(q_rot.shape, q.shape)
        self.assertEqual(k_rot.shape, k.shape)

    def test_no_nan(self):
        q = torch.randn(2, 4, self.seq_len, self.dim)
        k = torch.randn(2, 4, self.seq_len, self.dim)
        q_rot, k_rot = apply_rotary_emb(q, k, self.cos, self.sin)
        self.assertFalse(torch.isnan(q_rot).any())
        self.assertFalse(torch.isnan(k_rot).any())

    def test_position_zero_is_unchanged(self):
        # at position 0: cos=1, sin=0, so rotation is identity
        q = torch.randn(2, 4, self.seq_len, self.dim)
        k = torch.randn(2, 4, self.seq_len, self.dim)
        q_rot, k_rot = apply_rotary_emb(q, k, self.cos, self.sin)
        self.assertTrue(torch.allclose(q_rot[..., 0, :], q[..., 0, :], atol=1e-6))
        self.assertTrue(torch.allclose(k_rot[..., 0, :], k[..., 0, :], atol=1e-6))

    def test_norm_preserved(self):
        # RoPE is an orthogonal rotation, so ||q_rot|| == ||q||
        q = torch.randn(2, 4, self.seq_len, self.dim)
        k = torch.randn(2, 4, self.seq_len, self.dim)
        q_rot, k_rot = apply_rotary_emb(q, k, self.cos, self.sin)
        self.assertTrue(torch.allclose(q.norm(dim=-1), q_rot.norm(dim=-1), atol=1e-5))
        self.assertTrue(torch.allclose(k.norm(dim=-1), k_rot.norm(dim=-1), atol=1e-5))

    def test_relative_position_invariance(self):
        # The dot product q_i · k_j should equal q_{i+d} · k_{j+d} for any offset d,
        # because RoPE encodes relative position. We verify this for a single head.
        rope = RotaryEmbedding(self.dim)
        long_cos, long_sin = rope(self.seq_len + 5, device=torch.device('cpu'))

        q = torch.randn(1, self.seq_len, self.dim)
        k = torch.randn(1, self.seq_len, self.dim)

        # rotate at positions [0..seq_len]
        q_rot0, k_rot0 = apply_rotary_emb(q, k, long_cos[:self.seq_len], long_sin[:self.seq_len])
        # rotate at positions [5..seq_len+5]
        q_rot5, k_rot5 = apply_rotary_emb(q, k, long_cos[5:5 + self.seq_len], long_sin[5:5 + self.seq_len])

        # dot products between position i and i (i.e. relative offset = 0) should be equal regardless of absolute position
        dots0 = (q_rot0 * k_rot0).sum(dim=-1)  # (1, seq_len)
        dots5 = (q_rot5 * k_rot5).sum(dim=-1)  # (1, seq_len)
        self.assertTrue(torch.allclose(dots0, dots5, atol=1e-5))


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------

class TestAttentionNoMLM(unittest.TestCase):
    """Attention with mlm_mask_prob=0.0"""

    def setUp(self):
        self.attn = Attention(H, HEADS)
        self.x = torch.randn(B, R, C, H)

    def test_output_shape(self):
        out = self.attn(self.x)
        self.assertEqual(out.shape, (B, R, C, H))

    def test_no_nan(self):
        out = self.attn(self.x)
        self.assertFalse(torch.isnan(out).any())

    def test_with_attention_mask(self):
        mask = torch.ones(B, HEADS, R, C, C)
        mask[:, :, :, :, -1] = 0
        out = self.attn(self.x, mask)
        self.assertEqual(out.shape, (B, R, C, H))

    def test_deterministic_in_eval(self):
        self.attn.eval()
        out1 = self.attn(self.x)
        out2 = self.attn(self.x)
        self.assertTrue(torch.allclose(out1, out2))


# ---------------------------------------------------------------------------
# Axial_Transformer
# ---------------------------------------------------------------------------

class TestAxialTransformerNoMLM(unittest.TestCase):
    """Axial_Transformer with mlm_mask_prob=0.0"""

    def setUp(self):
        self.model = Axial_Transformer(H, HEADS)
        self.x = torch.randn(B, R, C, H)

    def test_output_shape(self):
        out = self.model(self.x)
        self.assertEqual(out.shape, (B, R, C, H))

    def test_no_nan(self):
        out = self.model(self.x)
        self.assertFalse(torch.isnan(out).any())

    def test_residual_modifies_input(self):
        out = self.model(self.x)
        self.assertFalse(torch.allclose(self.x, out))

    def test_deterministic_in_eval(self):
        self.model.eval()
        out1 = self.model(self.x)
        out2 = self.model(self.x)
        self.assertTrue(torch.allclose(out1, out2))

    def test_gradient_flow(self):
        x = self.x.requires_grad_(True)
        out = self.model(x)
        out.sum().backward()
        self.assertIsNotNone(x.grad)

# ---------------------------------------------------------------------------
# pair_matrix helper
# ---------------------------------------------------------------------------

class TestPairMatrix(unittest.TestCase):

    def test_shape(self):
        mat = pair_matrix(R, device=torch.device('cpu'))
        self.assertEqual(mat.shape, (NUM_PAIRS, R))

    def test_each_row_sums_to_two(self):
        mat = pair_matrix(R, device=torch.device('cpu'))
        row_sums = mat.sum(dim=1)
        self.assertTrue(torch.all(row_sums == 2))

    def test_values_are_zero_or_one(self):
        mat = pair_matrix(R, device=torch.device('cpu'))
        self.assertTrue(((mat == 0) | (mat == 1)).all())

    def test_distinct_pairs(self):
        mat = pair_matrix(R, device=torch.device('cpu'))
        # Each row should be unique
        rows = [tuple(mat[i].tolist()) for i in range(mat.size(0))]
        self.assertEqual(len(rows), len(set(rows)))


# ---------------------------------------------------------------------------
# PhyloLM
# ---------------------------------------------------------------------------

class TestPhyloLMNoMLM(unittest.TestCase):
    """PhyloLM with mlm_mask_prob=0.0"""

    def setUp(self):
        self.model = PhyloLM(
            num_rows=R, num_cols=C, num_blocks=2, h_dim=H,
            num_heads=HEADS, dropout=0.1
        )
        self.x = torch.randint(0, 21, (B, R, C))

    def test_output_shape(self):
        out = self.model(self.x)
        self.assertEqual(out.shape, (B, NUM_PAIRS))

    def test_no_nan(self):
        out = self.model(self.x)
        self.assertFalse(torch.isnan(out).any())

    def test_deterministic_in_eval(self):
        self.model.eval()
        out1 = self.model(self.x)
        out2 = self.model(self.x)
        self.assertTrue(torch.allclose(out1, out2))

    def test_gradient_flow(self):
        out = self.model(self.x)
        out.sum().backward()
        self.assertIsNotNone(self.model.embedding.weight.grad)

    def test_different_batch_sizes(self):
        for b in [1, 3]:
            x = torch.randint(0, 21, (b, R, C))
            out = self.model(x)
            self.assertEqual(out.shape, (b, NUM_PAIRS))

if __name__ == '__main__':
    unittest.main(verbosity=2)
