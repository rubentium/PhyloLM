
# PhyloLM: Reproducing and Extending Phyloformer

A high-performance reproduction of Phyloformer for likelihood-free phylogenetic inference, predicting pairwise evolutionary distances from Multiple Sequence Alignments (MSAs).

---

## Quick Start

### 1. Data Preprocessing
Before training, you must run the dataset through the `preprocess_memmaps.py` file to tokenize the datasets and turn them into `.dat` format. Run the following command:

```bash
python preprocess_memmaps.py --input_dir data/raw --output_dir data/processed
```

### 2. Training
The training process utilizes a custom dataloader located at `model/memmap_data.py`. This component is responsible for loading and batching the training and validation datasets from the memory-mapped files. To start training, run:

```bash
python train.py
```

### 3. Inference
To run inference, provide individual `.fasta` files to the `inference.py` file. The model will automatically utilize the weights found in the `checkpoints/` folder:

```bash
python inference.py --input path/to/your_sequence.fasta
```

---

## Repository Structure

* **`model/`**
    * `model.py`: Full Phyloformer++ architecture and prediction head.
    * **`memmap_data.py`**: Custom dataloader for `.dat` memory-mapped files.
    * `attention.py`: Axial attention modules and Flash Attention support.
    * `rope.py`: Rotary Positional Embedding (RoPE) utilities.
    * `preprocessing.py`: Tokenization and input preparation logic.
* **`train.py`**: Main entry point for the training loop and optimization.
* **`inference.py`**: Script for predicting distances from individual FASTA files.
* **`checkpoints/`**: Directory containing model checkpoints.

---

## Method & Architecture

PhyloLM reframes phylogenetic inference as a supervised regression problem. It predicts a pairwise evolutionary distance matrix directly from an MSA using an axial transformer architecture.

### Key Enhancements
* **ESM2 Tokenization:** Modern protein language model tokenization.
* **Learned Embeddings:** Captures biochemical similarity over simple one-hot encoding.
* **RoPE:** Preserves relative positional relationships across alignment sites.
* **Flash Attention:** Hardware-aware optimization for improved throughput.
* **Pre-Norm:** Increases optimization stability for deeper stacks.

### Pipeline Flow
1. **Tokenization:** Raw MSA is processed via ESM2 tokenizer ($S=50, L=500$).
2. **Pairwise Construction:** Constructs a representation for all $P = S(S-1)/2$ pairs.
3. **Axial Transformer:** Alternates attention across sites (columns) and pairs (rows).
4. **Distance Regression:** Predicts one scalar evolutionary distance per pair.

---

## Training Specifications

* **Dataset:** Trained on the **LG+GC** dataset (empirical amino acid replacement + heterogeneity).
* **Loss Function:** Mean Relative Error (MRE), normalizing error relative to branch depth.
* **Optimization:** Cosine annealing scheduler for smoother convergence.

| Parameters | Blocks | $d_{model}$ | Heads |
| :--- | :--- | :--- | :--- |
| **10.8M** | 10 | 256 | 16 |
| **2.7M** | 10 | 128 | 8 |
| **690K** | 10 | 64 | 4 |

---

## References
1. Nesterenko et al. *Phyloformer: Fast, accurate and versatile phylogenetic reconstruction.*
2. Dao et al. *FlashAttention: Fast and Memory-Efficient Exact Attention.*
3. Su et al. *RoFormer: Enhanced Transformer with Rotary Position Embedding.*
```
