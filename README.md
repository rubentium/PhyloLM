# PhyloLM: Reproducing and Extending Phyloformer

A high-performance reproduction of Phyloformer for likelihood-free phylogenetic inference, extended with modern transformer optimizations for improved stability, efficiency, and representational capacity.

This repository reproduces the core idea of predicting pairwise evolutionary distances directly from a multiple sequence alignment (MSA), while introducing architectural and systems-level improvements including **learned embeddings**, **ESM2 tokenization**, **RoPE**, **Flash Attention**, and **Pre-Norm transformer blocks**.

---

## Overview

Phylogenetic inference is traditionally posed as a combinatorial search over tree topologies, often requiring repeated likelihood evaluations. Phyloformer reframes this as a supervised regression problem: given an MSA, predict the corresponding pairwise evolutionary distance matrix directly.

This repository implements that pipeline and extends it with hardware-aware and transformer-based improvements.

```
MSA
  -> tokenization + learned embeddings
  -> pairwise representation construction
  -> axial transformer encoder
  -> distance regression
  -> pairwise evolutionary distances
  -> tree reconstruction (external tool)
```

---

## Repository Structure

```text
.
├── model/
│   ├── attention.py        # Axial attention modules, including Flash Attention support
│   ├── model.py            # Full Phyloformer++ architecture and prediction head
│   ├── preprocessing.py    # Tokenization, input preparation, tensor construction
│   ├── rope.py             # Rotary positional embedding utilities
│
├── trainer/
│   └── trainer.py          # Training loop, optimization, scheduling, validation
│
├── data/
│   ├── msas/               # Input alignments
│   └── trees/              # Ground-truth trees / distance supervision
│
├── README.md
└── requirements.txt
```

---

## Method

## Input Representation

Each input is a multiple sequence alignment with:

- `S = 50` sequences
- `L = 500` alignment positions

After tokenization with the **ESM2 tokenizer**, each sequence includes start and end tokens, producing an input array of shape:

```text
50 x 502
```

Unlike the original one-hot formulation, this implementation uses **learned embeddings**. This allows amino acids and special tokens to be represented in a continuous latent space rather than as equidistant categorical vectors.

### Why learned embeddings?

One-hot encoding imposes no notion of similarity between residues. Learned embeddings allow the model to capture:

- biochemical similarity
- substitutional tendencies
- context-dependent structure
- richer sequence representations

---

## Pairwise Representation Construction

The model does not operate directly on single-sequence features alone. Instead, it constructs a representation over **pairs of sequences**.

For `S = 50`, the number of unordered sequence pairs is:

```text
P = S(S - 1) / 2 = 1225
```

Thus, after pairwise construction, the model processes a tensor of shape:

```text
1225 x 502
```

This pairwise representation is central to the task, since the model predicts evolutionary distance for each sequence pair.

---

## Architecture

The model uses an **axial transformer architecture** designed to process pairwise MSA features efficiently.

### High-Level Architecture

```text
Input MSA: (S x L)
      |
      v
ESM2 Tokenizer
      |
      v
Learned Embedding Layer
      |
      v
Embedded MSA: (S x (L+2) x d_model)
      |
      v
Pairwise Tensor Construction
      |
      v
Pairwise Representation: (P x (L+2) x d_model)
      |
      v
Axial Transformer Stack
   - Pre-Norm row
   - site-wise / row-wise attention (with RoPE)
   - Pre-Norm col
   - pair-wise / column-wise attention
   - FFN
      |
      v
Distance Prediction Head
      |
      v
Predicted Distances: y_hat in R^P
```

---

## Axial Attention

To capture co-evolutionary signal efficiently, the model applies **alternating axial attention** rather than full 2D attention.

This means attention is factorized across relevant axes of the pairwise tensor, reducing cost while preserving useful structure.

### Components

- **Site-wise attention**: attends across alignment positions
- **Pair-wise / row-wise attention**: attends across sequence-pair dimension
- **Alternation across layers**: allows information to propagate across both structural directions

This design is especially useful for phylogenetic data, where signal arises both from local site patterns and from cross-sequence evolutionary relationships.

---

## Attention Equations

Let the input to an attention block be `X`.

The query, key, and value projections are:

```text
Q = X W_Q
K = X W_K
V = X W_V
```

Scaled dot-product attention is:

```text
Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V
```

For multi-head attention:

```text
head_i = Attention(Q_i, K_i, V_i)
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W_O
```

where:

- `d_k` is the key dimension per head
- `h` is the number of heads

---

## Rotary Positional Embeddings (RoPE)

For site-wise attention, absolute positional encodings are replaced with **Rotary Positional Embeddings (RoPE)**.

RoPE injects position by rotating query and key vectors in a way that preserves relative positional relationships.

### Motivation

In aligned protein sequences, relative positional structure matters more than arbitrary absolute indexing. RoPE is therefore a natural choice for modeling:

- conserved motifs
- shifted local structure
- long-range positional relationships across sites

Conceptually, RoPE modifies `Q` and `K` before the attention score is computed:

```text
Q_rot = RoPE(Q)
K_rot = RoPE(K)
Attention = softmax(Q_rot K_rot^T / sqrt(d_k)) V
```

---

## Flash Attention

The model integrates **Flash Attention** to improve memory efficiency and throughput.

Instead of materializing large attention matrices in standard form, Flash Attention computes exact attention with IO-aware tiling, reducing memory traffic between fast and slow memory.

### Why this matters

Phylogenetic transformer models are memory-intensive. Flash Attention enables:

- larger effective model sizes
- higher throughput
- improved GPU utilization
- reduced memory bottlenecks

In this implementation, the **head dimension is fixed at 16**, chosen to align well with efficient attention kernel execution.

---

## Pre-Norm Transformer Blocks

This implementation uses a **Pre-Norm** transformer configuration rather than Post-Norm.

### Pre-Norm block structure

```text
x -> LayerNorm -> Attention -> Residual Add
x -> LayerNorm -> Feedforward -> Residual Add
```

This improves optimization stability, especially in deeper transformer stacks, by preserving a cleaner gradient path through the residual branch.

---

## Full Model (`model.py`)

The full architecture combines:

- ESM2 tokenization
- learned embeddings
- pairwise representation construction
- alternating axial attention
- RoPE for site-wise attention
- Flash Attention
- Pre-Norm transformer blocks
- distance regression head

The final output is a vector of predicted pairwise distances:

```text
y_hat in R^P
```

where:

```text
P = S(S - 1) / 2
```

For `S = 50`:

```text
P = 1225
```

---

## File-Level Description

### `model/preprocessing.py`

Prepares raw biological input for the model.

Responsibilities may include:

- loading MSAs
- tokenizing sequences with the ESM2 tokenizer
- inserting start/end tokens
- constructing tensor inputs
- organizing targets for pairwise distance supervision

Typical transformations:

```text
raw MSA
  -> tokenized sequences
  -> embedded input indices
  -> pairwise construction
  -> model-ready tensors
```

---

### `model/attention.py`

Implements the attention machinery used by the model.

Responsibilities may include:

- multi-head attention
- axial attention over chosen tensor dimensions
- Flash Attention integration
- attention reshaping and projection logic
- support for head dimension configuration

This file is the core computational component of the architecture.

---

### `model/rope.py`

Implements Rotary Positional Embedding utilities.

Responsibilities may include:

- computing sinusoidal rotation factors
- applying rotary transformations to queries and keys
- handling positional indexing for site-wise attention

This allows relative positional structure to be encoded without relying on fixed absolute embeddings.

---

### `model/model.py`

Defines the end-to-end network.

Responsibilities may include:

- embedding lookup / projection
- pairwise tensor construction
- transformer block stacking
- normalization and residual flow
- final distance prediction layer

This file connects all model components into the final regression pipeline.

---

### `trainer/trainer.py`

Implements optimization and training logic.

Responsibilities may include:

- training step computation
- validation loop
- metric tracking
- gradient accumulation
- optimizer and scheduler setup
- checkpointing

This file is responsible for running experiments consistently across model sizes.

---

## Training Objective

The primary objective used in this project is **Mean Relative Error (MRE)**.

Given true distances `d_i` and predicted distances `d_hat_i`, the loss is:

```text
L_MRE = (1 / N) * sum_{i=1}^N |(d_hat_i - d_i) / d_i|
```

This objective is well-suited for phylogenetics because it normalizes error relative to branch depth, preventing large-magnitude distances from dominating optimization.

### Alternative loss

A standard absolute regression objective can also be written as Mean Absolute Error (MAE):

```text
L_MAE = (1 / N) * sum_{i=1}^N |d_hat_i - d_i|
```

In this project, **MRE** is emphasized as the more appropriate metric for phylogenetic distance prediction.

---

## Dataset

The model is trained on the **LG+GC** dataset.

### LG+GC components

- **LG**: empirical amino acid replacement matrix
- **GC**: accounts for site-rate variation and compositional heterogeneity

This dataset provides simulated protein evolution data suitable for supervised learning of phylogenetic distances.

---

## Training Setup

Training was standardized to:

- **120,000 total samples**
- **5,000 optimization steps**
- **24 samples per step**

Because memory requirements vary across model sizes, the implementation balances:

- batch size
- gradient accumulation steps

so that each update processes the same effective number of datapoints.

### Optimization

Training uses a **cosine annealing scheduler**.

This produces a learning-rate trajectory of the form:

```text
eta_t = eta_min + 0.5 * (eta_max - eta_min) * (1 + cos(pi t / T))
```

where:

- `eta_t` is the learning rate at step `t`
- `T` is the total schedule length

This allows:

- rapid exploration early in training
- smoother convergence later in training

---

## Training Command

Training is run through:

```text
trainer/trainer.py
```

Example command:

```bash
python trainer/trainer.py \
    --train-msas data/train/msas \
    --train-trees data/train/trees \
    --val-msas data/val/msas \
    --val-trees data/val/trees \
    --scheduler cosine \
    --loss mre \
    --steps 5000
```

If your argument names differ, update the flags accordingly, but keep `trainer/trainer.py` as the entry point.

---

## Example End-to-End Pipeline

### Step 1: Input

Start with an MSA of `S = 50` aligned protein sequences of length `L = 500`.

### Step 2: Tokenization

Apply the ESM2 tokenizer and add start/end tokens:

```text
50 x 500 -> 50 x 502
```

### Step 3: Embedding

Map tokens into a learned embedding space:

```text
(50 x 502) -> (50 x 502 x d_model)
```

### Step 4: Pairwise Construction

Construct all unordered sequence pairs:

```text
P = 50 * 49 / 2 = 1225
```

giving:

```text
(1225 x 502 x d_model)
```

### Step 5: Axial Transformer

Process the pairwise tensor with alternating axial attention, using:

- Flash Attention
- RoPE
- Pre-Norm blocks

### Step 6: Regression

Predict one scalar evolutionary distance per pair:

```text
y_hat in R^1225
```

### Step 7: Tree Reconstruction

Pass the predicted distance matrix to an external tool such as:

- FastME
- FastTree
- IQTree

to obtain a phylogenetic tree.

---

## Model Configurations

The following model scales were studied:

| Parameters | Blocks | d_model | Heads |
|-----------:|-------:|--------:|------:|
| 10.8M      | 10     | 256     | 16    |
| 2.7M       | 10     | 128     | 8     |
| 690K       | 10     | 64      | 4     |
| 179K       | 10     | 32      | 2     |

These comparisons suggest that both **depth** and **model width** play important roles in phylogenetic signal extraction.

---

## Reproduction Scope

This repository reproduces the central learning formulation of Phyloformer:

- MSA-based distance prediction
- transformer-based phylogenetic regression
- likelihood-free inference pipeline

At the same time, it introduces several principled extensions:

- learned embeddings in place of one-hot encodings
- ESM2 tokenization
- RoPE for positional structure
- Flash Attention for efficiency
- Pre-Norm for stability
- cosine annealing for training dynamics

---

## Differences from Original Phyloformer

Compared with the original formulation, this implementation introduces several major modifications:

### 1. Learned Embeddings
The original one-hot input encoding is replaced by learned embeddings.

### 2. ESM2 Tokenization
The input pipeline is adapted to a modern protein language modeling tokenizer.

### 3. RoPE
Rotary positional embeddings replace absolute positional encodings for site-wise attention.

### 4. Flash Attention
Attention computation is upgraded for improved throughput and reduced memory usage.

### 5. Pre-Norm
Layer normalization is moved before each sub-layer to improve optimization stability.

### 6. Hardware-Aware Design
The implementation is tuned for efficient execution on modern accelerators.

---

## Results Summary

Across tested configurations, the **10.8M parameter model** achieved the strongest and most stable validation performance in predicting evolutionary distances.

The comparison between the `690K` and `430K` models also highlights the importance of **depth**: the 10-block variant substantially outperformed the 6-block variant, indicating that deeper transformer stacks are important for resolving complex site-wise dependencies.

---

## Notes

- This repository emphasizes clarity, reproducibility, and architectural transparency.
- The implementation is designed for efficient training on modern GPUs.
- The project serves both as a reproduction study and as a platform for architectural extension.

---

## References

1. Nesterenko, L., Blassel, L., Veber, P., Boussau, B., and Jacob, L. Phyloformer: Fast, accurate and versatile phylogenetic reconstruction with deep neural networks.
2. Vaswani, A. et al. Attention Is All You Need.
3. Xiong, R. et al. On Layer Normalization in the Transformer Architecture.
4. Su, J. et al. RoFormer: Enhanced Transformer with Rotary Position Embedding.
5. Dao, T. et al. FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness.
6. Lin, Z. et al. Evolutionary-Scale Prediction of Atomic-Level Protein Structure with a Language Model.
