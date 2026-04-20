# PhyloLM

PhyloLM predicts pairwise patristic distances directly from a multiple sequence alignment (MSA). The code follows the core Phyloformer idea of moving from taxa to pair representations and then applying axial attention over the pair and site dimensions, but this implementation adds several engineering choices aimed at making training practical on modern GPUs: ESM2 tokenization, memory-mapped datasets, `torch.compile`, bfloat16 execution, PyTorch SDPA for dense attention, and a block-sparse row-attention path built on `flex_attention`.

## What The Model Predicts

For an alignment with `R` sequences, the model predicts one scalar for each unordered pair of taxa:

`P = R * (R - 1) / 2`

Those targets are patristic distances computed from the matching Newick tree. With the current dataset conventions in this repository, `R = 50`, so `P = 1225`.

## Data Pipeline

### Expected raw files

The preprocessing code expects one alignment and one tree per sample, matched by filename stem:

- `{id}_50_tips.fasta`
- `{id}_50_tips.nwk`

The FASTA and tree files are paired by the shared `{id}` prefix. Unmatched files are dropped.

### Tokenization

Tokenization is handled by the Hugging Face tokenizer for `facebook/esm2_t6_8M_UR50D`.

- Implementation: `Tokenizer` in [data.py](./data.py)
- Backend: `transformers.AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")`
- Output: integer token ids shaped `(R, C)` for each alignment
- Vocabulary size: 33 tokens, which is why alignment memmaps are stored as `int8`

The FASTA parser preserves sequence order, and that same order is used when patristic distances are extracted from the Newick tree. That matters because the target vector is built over all unordered pairs in FASTA order.

### Tree target creation

For each `.nwk` file, the preprocessing code:

1. Parses the tree with `dendropy`
2. Builds a phylogenetic distance matrix
3. Extracts pairwise distances for all pairs of taxa in FASTA order
4. Stores the result as a flat vector of length `P`

The pair ordering matches `torch.combinations(torch.arange(rows), r=2)` in the model, so targets and predictions line up exactly.

### `.dat` file creation

Run preprocessing to convert raw FASTA/Newick pairs into memory-mapped binary files:

```bash
python preprocess_memmaps.py \
  --train_alignment_dir /path/to/train/alignments \
  --train_tree_dir /path/to/train/trees \
  --val_alignment_dir /path/to/val/alignments \
  --val_tree_dir /path/to/val/trees \
  --output_dir /path/to/LG_GC_memmaps
```

This writes the following structure:

```text
LG_GC_memmaps/
  train/
    alignments.dat
    trees.dat
    meta.json
  val/
    alignments.dat
    trees.dat
    meta.json
```

On-disk formats:

- `alignments.dat`: `int8`, shape `(N, R, C)`
- `trees.dat`: `int16`, shape `(N, P)`
- `meta.json`: split metadata and tensor shapes

`trees.dat` is not plain `int16` data semantically. The distances are cast to `bfloat16`, then reinterpreted as raw `int16` bits before being written. At load time they are restored with:

```python
distances = torch.from_numpy(trees_np).view(torch.bfloat16)
```

That keeps the target file compact while still using bfloat16 during training.

### Runtime loading

Training does not use the eager in-memory `PhyloDataset` path by default. The actual training script uses the memmap iterator in [model/memmap_data.py](./model/memmap_data.py).

Key details:

- Reads whole batches directly from the memmaps
- Converts alignment tokens to `torch.int64`
- Reinterprets tree targets as `torch.bfloat16`
- Optionally pins memory for async GPU transfer
- Filters corrupted all-zero samples once at startup
- Prefetches with a background thread instead of multiprocessing workers

That thread-based prefetching is deliberate: the code is written to avoid `/dev/shm` pressure from multiprocessing data loaders.

## Architecture

### High-level structure

The model is implemented in [model/model.py](./model/model.py) and [model/axial_transfomer.py](./model/axial_transfomer.py).

Pipeline:

1. Token ids `(B, R, C)` are embedded to `(B, R, C, H)`
2. The tensor is projected from taxa space into pair space with a fixed pair-incidence matrix
3. The pair representation is processed by a stack of axial transformer blocks
4. Two small output heads reduce `(B, num_pairs, C, H)` to `(B, num_pairs)`

The key structural move is the pair projection:

- `pair_matrix_f(rows)` builds a binary matrix with one row per taxon pair
- Multiplying by that matrix converts per-taxon embeddings into per-pair embeddings
- This is the same modeling viewpoint that makes Phyloformer attractive for phylogenetic distance prediction

### Axial attention layout

Each block alternates attention along two axes:

- Row attention: across pair tokens for each alignment column
- Column attention: across alignment columns for each pair token

This keeps attention factorized instead of paying the full cost of 2D attention over pair-by-site tokens.

### Normalization and block layout

Each axial block uses pre-norm residual structure with `nn.RMSNorm`:

- `row_norm` before row attention
- `col_norm` before column attention
- `ffn_norm` before the feed-forward network

The feed-forward path is:

```text
Linear(H, 4H) -> GELU -> Linear(4H, H)
```

The final readout is:

```text
Linear(H, 4H) -> GELU -> Linear(4H, 1) -> Linear(C, 1)
```

### Dense attention path: SDPA and Flash Attention

Dense attention uses `torch.nn.functional.scaled_dot_product_attention`.

Important implementation detail:

- The attention tensors are naturally 5D because of the axial layout: `(B, extra, heads, seq, dim)`
- The code flattens the leading two dimensions to `(B * extra, heads, seq, dim)` before calling SDPA
- That is done specifically so PyTorch can dispatch to the fused Flash or memory-efficient kernels when available
- The head dimension needs to be a multiple of 16 for Flash Attention (part of SDPA) to automatically kick in, else fallback to slow attention

This is the dense path used by `att_type=dense` and also by the sparse model when `full_att=True`.

### Sparse attention path

Sparse row attention is implemented in [model/sparse_attention.py](./model/sparse_attention.py) with `torch.nn.attention.flex_attention`.

The sparse pattern is block-based:

- Sequence length is divided into blocks of size 128
- Each query block always attends to its own diagonal block
- It also attends to `num_random_blocks` off-diagonal blocks sampled without replacement

Implementation details worth knowing:

- A pool of 1000 block masks is built at initialization time
- Training samples a mask index from a dedicated NumPy RNG stream
- Evaluation uses mask 0 for deterministic behavior
- `full_att=True` bypasses the sparse mask and runs dense SDPA on the valid, non-padded region

### RoPE

Rotary positional embeddings are implemented in [model/rope.py](./model/rope.py).

- RoPE is applied to the column-attention queries and keys
- RoPE is not applied to row attention over pair tokens

That choice is consistent with the semantics of the axes: columns are ordered sequence positions, while pair rows are combinatorial objects rather than a natural 1D order.

### Padding for sparse attention

The sparse implementation currently pads pair rows from 1225 to 1280 using:

- top padding: 27
- bottom padding: 28

That is done so the pair axis tiles cleanly into 10 blocks of size 128 for `flex_attention`.

This means the current sparse path is effectively tuned to the 50-tip setting used here:

- valid pairs: `C(50, 2) = 1225`
- padded pairs: `1280 = 10 * 128`

Dense attention has no such padding requirement.

## Relationship To Phyloformer

This repository is based on the Phyloformer modeling idea rather than being a line-for-line reproduction of the paper.

The shared core idea is:

- start from an MSA
- build representations for taxon pairs rather than single taxa only
- apply axial attention over pair and site dimensions
- predict pairwise evolutionary distances directly

The main implementation choices here that are more engineering-driven than paper-faithful are:

- ESM2 tokenizer instead of a hand-rolled alphabet tokenizer
- RMSNorm everywhere in the transformer blocks
- PyTorch SDPA for dense attention so Flash/memory-efficient kernels can be used
- A block-sparse row-attention variant using `flex_attention`
- bfloat16 training and target storage
- memory-mapped datasets for larger-scale training

## Training

Install dependencies first:

```bash
pip install -r requirements.txt
```

Then train from the memmap dataset:

```bash
python train.py --memmap_dir /path/to/LG_GC_memmaps
```

Useful training flags:

```bash
python train.py \
  --memmap_dir /path/to/LG_GC_memmaps \
  --att_type sparse \
  --num_blocks 10 \
  --h_dim 128 \
  --num_heads 8 \
  --num_random_blocks 1 \
  --batch_size 1 \
  --grad_accum_steps 48 \
  --criterion mae \
  --use_wandb
```

Training details from the implementation:

- The model is moved to `torch.bfloat16`
- Forward passes run under CUDA autocast in bfloat16
- The model is wrapped with `torch.compile`
- Optimizer options are `adam`, `adamw`, or `sgd`
- The scheduler is linear warmup followed by cosine decay
- Validation can optionally compare sparse attention against dense full attention
- Checkpoints are written into `checkpoints/`

Default training entry point assumptions:

- `train.py` expects a prebuilt memmap directory, not raw FASTA/Newick files
- Dense and sparse models share the same high-level code path
- Sparse training samples one random block mask per layer per forward pass

## Inference

Run inference on a single MSA with:

```bash
python inference.py \
  --fasta_path /path/to/sample_50_tips.fasta \
  --checkpoint checkpoints/final_checkpoint_*.pt \
  --output distances.pt
```

The inference script:

- tokenizes the input FASTA with the same ESM2 tokenizer
- rebuilds a `PhyloLM` model from the input shape
- loads checkpoint weights after stripping the `_orig_mod.` prefix added by `torch.compile`
- saves the predicted pairwise distances as a PyTorch tensor

One caveat: the current inference script hardcodes the architecture to 10 blocks, hidden size 128, 8 heads, and the model default attention type. If you trained a checkpoint with different hyperparameters, update [inference.py](./inference.py) accordingly before loading it.

## Profiling

There is also a simple profiler entry point:

```bash
python profiler.py --memmap_dir /path/to/LG_GC_memmaps
```

It performs a compile warmup, then captures CPU and CUDA traces with `torch.profiler`.

## Tests

The test suite focuses mainly on the sparse attention implementation:

- [tests/test_build_global_mask.py](./tests/test_build_global_mask.py)
- [tests/test_sparse_vs_dense.py](./tests/test_sparse_vs_dense.py)

These tests check mask construction, random-block selection, sparse-vs-dense behavioral differences, and the `full_att` bypass path.

## Practical Notes

- Sparse attention requires CUDA and PyTorch support for `flex_attention`.
- Dense attention is simpler to run, but the repository is clearly optimized for GPU execution
- The current sparse padding logic is specialized for 50-sequence alignments
- Preprocessing and training both assume the FASTA leaf order and Newick leaf labels match exactly

#### Author: Ruben Navasardyan, UWaterloo/EPFL