# PhyloLM: Reproducing and Extending Phyloformer

A Transformer-based framework for learning pairwise evolutionary distances from multiple sequence alignments, reproducing and extending the Phyloformer architecture.

This repository contains a **clean reimplementation and extension** of the Phyloformer model introduced in:

> Nesterenko et al. (2025), *Phyloformer: Fast, accurate and versatile phylogenetic reconstruction with deep neural networks*

Our objective is twofold:
- **Reproduce** the Phyloformer pipeline from first principles  
- **Extend** the framework toward scalable and generalizable representations  

---

## Overview

Phylogenetic inference aims to reconstruct evolutionary relationships from a **multiple sequence alignment (MSA)**.

Phyloformer replaces traditional likelihood-based methods with a neural network that directly predicts **pairwise evolutionary distances**:

```
MSA → Neural Network → Distance Matrix → Tree (external tool)
```

This repository implements the **MSA → distance prediction** stage.

---

## Repository Structure

```
.
├── model/
│   ├── attention.py        # Multi-head self-attention
│   ├── model.py            # Full architecture
│   ├── preprocessing.py    # MSA → tensor pipeline
│   ├── rope.py             # Positional encoding (optional)
│
├── trainer/
│   └── trainer.py          # Training loop and optimization
│
├── data/
│   ├── msas/
│   └── trees/
│
├── README.md
└── requirements.txt
```

---

## Architecture

The model follows a **Transformer-style architecture adapted to structured biological sequences**.

---

### High-Level Diagram

```
MSA (S × L × 22)
        │
        ▼
Embedding / Linear Projection
        │
        ▼
┌───────────────────────────────┐
│   Attention Block × N         │
│   - Multi-head attention      │
│   - Feedforward layer         │
└───────────────────────────────┘
        │
        ▼
Pairwise Feature Aggregation
        │
        ▼
Distance Prediction Head
        │
        ▼
Output: vector of size P = S(S-1)/2
```

---

### Detailed Architecture (Expanded View)

```
                ┌────────────────────────────┐
                │        Input MSA           │
                │      (S × L × 22)          │
                └────────────┬───────────────┘
                             │
                             ▼
                ┌────────────────────────────┐
                │ Linear Projection Layer    │
                │ (22 → embed_dim)           │
                └────────────┬───────────────┘
                             │
                             ▼
        ┌────────────────────────────────────────────┐
        │         Transformer Encoder Stack          │
        │                                            │
        │  ┌──────────────────────────────────────┐  │
        │  │ Multi-Head Attention                │  │
        │  │ Q, K, V projections                │  │
        │  │ Scaled dot-product attention       │  │
        │  └──────────────────────────────────────┘  │
        │                    │                       │
        │  ┌──────────────────────────────────────┐  │
        │  │ Feedforward Network                 │  │
        │  │ (position-wise MLP)                 │  │
        │  └──────────────────────────────────────┘  │
        │                                            │
        │        (repeated N times)                  │
        └────────────┬───────────────────────────────┘
                     │
                     ▼
        ┌────────────────────────────────────────────┐
        │ Pairwise Representation Construction       │
        │ Combine sequence embeddings (i, j)         │
        └────────────┬───────────────────────────────┘
                     │
                     ▼
        ┌────────────────────────────────────────────┐
        │ Distance Prediction Head                   │
        │ (MLP or linear projection)                │
        └────────────┬───────────────────────────────┘
                     │
                     ▼
        ┌────────────────────────────────────────────┐
        │ Output: Pairwise Distances                │
        │ Size: P = S(S−1)/2                        │
        └────────────────────────────────────────────┘
```

### Input Representation

- S: number of sequences  
- L: alignment length  
- Each residue is encoded as a **22-dimensional one-hot vector**

```
x has shape (S, L, 22)
```

---

## Core Components

### Attention Module (`attention.py`)

Implements multi-head self-attention over MSA tensors.

**Input**
```
(batch_size, S, L, embed_dim)
```

**Computation**

```
Q = x W_Q
K = x W_K
V = x W_V
```

```
Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V
```

```
head_i = Attention(Q_i, K_i, V_i)
MultiHead = Concat(head_1, ..., head_h) W_O
```

**Output**
```
Same shape as input
```

This allows the model to capture:
- inter-sequence relationships  
- positional dependencies within alignments  

---

### Full Model (`model.py`)

Defines the complete architecture.

**Components**
- Input projection layer  
- Stack of attention blocks  
- Optional positional encoding (RoPE)  
- Final prediction head  

**Output**
```
y has length P = S * (S - 1) / 2
```

This corresponds to the **upper triangle of the distance matrix**.

---

### Preprocessing (`preprocessing.py`)

Responsible for transforming biological data into model inputs.

**Pipeline**
1. Read FASTA alignment  
2. Convert amino acids → indices  
3. One-hot encode sequences  
4. Load tree file (Newick format)  
5. Compute pairwise evolutionary distances  

**Key functions**
- `read_fasta`
- `msa_to_indices`
- `one_hot_encode`
- `load_alignment`
- `load_distance_matrix`

**Output**
```
x: (S, L, 22)
y: (P,)
```

---

### Training (`trainer.py`)

Implements optimization and training logic.

**Loss Functions**

Mean Absolute Error:
```
MAE = (1 / P) * sum |y_pred - y_true|
```

Mean Relative Error:
```
MRE = (1 / P) * sum |y_pred - y_true| / (y_true + epsilon)
```

**Training Features**
- batching over MSAs  
- validation evaluation  
- modular loss selection  
- extensible training loop  

---

## Data Format

```
data/
├── train/
│   ├── msas/
│   └── trees/
└── val/
    ├── msas/
    └── trees/
```

Each alignment must match its tree:

```
msas/0_20_tips.fa  ↔  trees/0_20_tips.nwk
```

---

## Running the Model

### Preprocessing
```
x, y = load_alignment(msa_path, tree_path)
```

### Forward Pass
```
x → (1, S, L, 22)
model(x) → predicted distances
```

### Tree Reconstruction

Use external tools:
- FastME  
- FastTree  
- IQTree  

---

## Example: End-to-End Pipeline

This section illustrates the full data flow through the model.

### Step 1: Input MSA

Example alignment (simplified):

```
Seq1: A R N D
Seq2: A R N -
Seq3: A - N D
```

### Step 2: One-hot Encoding

Each amino acid is mapped to a 22-dimensional vector:

```
A → [1, 0, 0, ..., 0]
R → [0, 1, 0, ..., 0]
...
- → gap token
```

Resulting tensor:

```
x shape: (S=3, L=4, 22)
```

---

### Step 3: Model Forward Pass

```
x → embedding → attention blocks → pairwise features → distances
```

---

### Step 4: Output

Model predicts pairwise distances:

```
y_pred = [
    d(Seq1, Seq2),
    d(Seq1, Seq3),
    d(Seq2, Seq3)
]
```

This corresponds to:

```
P = S(S−1)/2 = 3
```

---

### Step 5: Tree Reconstruction

Distances can be passed to a phylogenetic tool:

```
distance matrix → FastME → tree (.nwk)
```

## Training

Training is handled through:

```
trainer/trainer.py
```

### Example Usage

```
python trainer/trainer.py \
    --train-msas data/train/msas \
    --train-trees data/train/trees \
    --val-msas data/val/msas \
    --val-trees data/val/trees \
    --epochs 20 \
    --batch-size 4 \
    --learning-rate 1e-4
```

### Key Parameters

- `--epochs`: number of training epochs  
- `--batch-size`: batch size  
- `--learning-rate`: optimizer step size  

### Notes

- GPU recommended for training  
- Large MSAs may require smaller batch sizes  
- Validation helps detect overfitting  

---

## Reproduction Scope

This implementation reproduces:
- MSA encoding pipeline  
- Transformer-based architecture  
- Distance prediction objective  

Design principles:
- No direct code reuse from original repository  
- Minimal, interpretable implementation  
- Focus on correctness and reproducibility  

---

## Differences from Original Phyloformer

This implementation differs from the original repository in several key aspects:

### 1. Clean Reimplementation
- No direct reuse of original source code  
- All components rewritten from first principles  
- Emphasis on clarity and interpretability  

### 2. Simplified Architecture
- Minimal attention implementation  
- Reduced engineering complexity  
- Focus on core modeling behavior  

### 3. Modular Design
- Clear separation between preprocessing, model, and training  
- Easier to extend and experiment  

### 4. Reproducibility-Oriented
- Transparent data pipeline  
- Explicit tensor shapes and transformations  
- Designed for educational and research use  

### 5. Extension-Ready
- Structured to support:
  - alternative embeddings  
  - scaling experiments  
  - integration with modern representation learning methods
  - 
## Extensions

We explore several directions beyond the original work:

- Learned embeddings replacing one-hot encoding  
- Scaling model capacity  
- Efficiency improvements  
- Generalization toward representation learning  

---

## Notes

- Designed for clarity rather than production optimization  
- Modular structure for experimentation  
- Compatible with CPU (GPU preferred)  

---

## References

- Nesterenko et al. (2025), *Phyloformer*
