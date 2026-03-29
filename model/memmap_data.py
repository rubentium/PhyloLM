"""
model/memmap_data.py

Low-latency batch iterator that reads from the np.memmap binary files
produced by preprocess_memmaps.py.

Design constraints:
    - /dev/shm is only 64 MB → no multiprocessing DataLoader workers.
      All prefetching is done via a single background *thread* (shares
      the process heap, no IPC serialization).
    - Corrupted all-zero samples are detected at init and excluded.
    - Whole batches are sliced from the memmap in one vectorized read,
      then dtype-converted and pin_memory()'d for async GPU transfer.

Each batch yields:
    alignment  : LongTensor      (B, R, C)  — token ids
    distances  : BFloat16Tensor  (B, P)     — patristic distances

The on-disk layout is:
    <memmap_dir>/
        train/
            alignments.dat    int8   (N, R, C)
            trees.dat         int16  (N, P)   <- raw bfloat16 bits
            meta.json
        val/   (same structure)
"""

import os
import json
import logging
import threading
import queue

import numpy as np
import torch

logger = logging.getLogger(__name__)

_PIN = torch.cuda.is_available()

class MemmapBatchIterator:
    """
    Reads whole batches directly from np.memmap files, converts dtypes in
    bulk, and optionally pins memory for non_blocking GPU transfer.

    Arguments:
        split_dir  : directory with meta.json, alignments.dat, trees.dat
        batch_size : samples per batch
        shuffle    : whether to shuffle indices each epoch
        drop_last  : drop the final incomplete batch (recommended for train)
        cycle      : automatically restart (and reshuffle) at epoch end
        pin_memory : call .pin_memory() on returned tensors
        seed       : RNG seed for reproducible shuffling
    """

    def __init__(
        self,
        split_dir: str,
        batch_size: int = 32,
        shuffle: bool = False,
        drop_last: bool = False,
        cycle: bool = False,
        pin_memory: bool = _PIN,
        seed: int = 42,
    ):
        meta_path = os.path.join(split_dir, "meta.json")
        if not os.path.isfile(meta_path):
            raise FileNotFoundError(
                f"meta.json not found in {split_dir!r}. "
                "Run preprocess_memmaps.py first."
            )

        with open(meta_path) as f:
            meta = json.load(f)

        N = meta["num_samples"]
        R = meta["num_seqs"]
        C = meta["seq_len"]
        P = meta["num_pairs"]

        align_path = os.path.join(split_dir, meta["alignments"]["file"])
        trees_path = os.path.join(split_dir, meta["trees"]["file"])

        self._alignments = np.memmap(align_path, dtype=np.int8, mode="r", shape=(N, R, C))
        self._trees = np.memmap(trees_path, dtype=np.int16, mode="r", shape=(N, P))

        # --- filter out corrupted (all-zero) samples ----------------------
        # Check in chunks to avoid loading the entire memmap into RAM at once.
        CHUNK = 4096
        valid_mask = np.ones(N, dtype=bool)
        for start in range(0, N, CHUNK):
            end = min(start + CHUNK, N)
            chunk = np.array(self._alignments[start:end])  # (chunk, R, C) copy
            # A sample is corrupt if every byte is zero
            all_zero = ~chunk.any(axis=(1, 2))
            valid_mask[start:end] = ~all_zero

        self._valid_indices = np.where(valid_mask)[0]
        n_filtered = N - len(self._valid_indices)
        if n_filtered:
            logger.warning(
                "Filtered %d / %d corrupted (all-zero) samples in %s",
                n_filtered, N, split_dir,
            )

        # public metadata
        self.num_rows = R
        self.num_cols = C
        self.num_pairs = P
        self.num_samples = len(self._valid_indices)

        self._batch_size = batch_size
        self._shuffle = shuffle
        self._drop_last = drop_last
        self._cycle = cycle
        self._pin = pin_memory
        self._rng = np.random.default_rng(seed)

        # prepare first epoch
        self._order = self._valid_indices.copy()
        if self._shuffle:
            self._rng.shuffle(self._order)
        self._pos = 0

    def _reset(self):
        """Reshuffle (if needed) and rewind to the start of the epoch."""
        self._order = self._valid_indices.copy()
        if self._shuffle:
            self._rng.shuffle(self._order)
        self._pos = 0

    def __iter__(self):
        self._reset()
        return self

    def __next__(self):
        while True:
            remaining = len(self._order) - self._pos
            if remaining <= 0 or (self._drop_last and remaining < self._batch_size):
                if self._cycle:
                    self._reset()
                    continue
                raise StopIteration

            end = min(self._pos + self._batch_size, len(self._order))
            indices = self._order[self._pos:end]
            self._pos = end

            # --- bulk memmap read + dtype conversion ----------------------
            # np.array() copies the slice out of the memmap in one read
            align_np = np.array(self._alignments[indices])   # (B, R, C) int8
            trees_np = np.array(self._trees[indices])         # (B, P)    int16

            alignment = torch.from_numpy(align_np.astype(np.int64))
            distances = torch.from_numpy(trees_np).view(torch.bfloat16)

            if self._pin:
                alignment = alignment.pin_memory()
                distances = distances.pin_memory()

            return alignment, distances

    def __len__(self):
        """Number of batches per epoch."""
        n = self.num_samples
        if self._drop_last:
            return n // self._batch_size
        return (n + self._batch_size - 1) // self._batch_size


class PrefetchIterator:
    """
    Wraps any iterator and fetches ahead on a daemon thread.
    Uses a bounded queue so at most *bufsize* batches sit in CPU memory.

    Because this is thread-based (not process-based), tensors live on the
    regular process heap — /dev/shm is never touched.
    """

    _SENTINEL = object()

    def __init__(self, base_iter, bufsize: int = 2):
        self._base = base_iter
        self._queue: queue.Queue = queue.Queue(maxsize=bufsize)
        self._thread: threading.Thread | None = None
        # expose metadata from the underlying iterator
        if hasattr(base_iter, "num_rows"):
            self.num_rows = base_iter.num_rows
        if hasattr(base_iter, "num_cols"):
            self.num_cols = base_iter.num_cols
        if hasattr(base_iter, "num_pairs"):
            self.num_pairs = base_iter.num_pairs
        if hasattr(base_iter, "num_samples"):
            self.num_samples = base_iter.num_samples

    def _producer(self):
        try:
            for item in self._base:
                self._queue.put(item)
        except Exception as exc:
            self._queue.put(exc)
        finally:
            self._queue.put(self._SENTINEL)

    def __iter__(self):
        # restart the underlying iterator (triggers reshuffle if applicable)
        iter(self._base)
        self._thread = threading.Thread(target=self._producer, daemon=True)
        self._thread.start()
        return self

    def __next__(self):
        item = self._queue.get()
        if item is self._SENTINEL:
            self._thread.join()
            raise StopIteration
        if isinstance(item, Exception):
            self._thread.join()
            raise item
        return item

    def __len__(self):
        return len(self._base)


def create_memmap_dataloaders(
    memmap_dir: str,
    batch_size: int = 32,
    seed: int = 42,
    prefetch: int = 2,
):
    """
    Build train and val batch iterators from a memmap directory produced by
    preprocess_memmaps.py.

    All IO happens via a background thread — no multiprocessing workers
    and no /dev/shm usage (safe with 64 MB shm).

    Arguments:
        memmap_dir : root directory containing ``train/`` and ``val/`` sub-dirs.
        batch_size : samples per batch.
        seed       : RNG seed for reproducible shuffling.
        prefetch   : number of batches to prefetch (0 disables prefetching).

    Returns:
        (train_iter, val_iter) — both are infinite cycling iterators.
        Access .num_rows, .num_cols, .num_samples on either for metadata.
    """
    train_iter = MemmapBatchIterator(
        os.path.join(memmap_dir, "train"),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        cycle=True,
        seed=seed,
    )
    val_iter = MemmapBatchIterator(
        os.path.join(memmap_dir, "val"),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        cycle=True,
        seed=seed,
    )

    if prefetch > 0:
        train_iter = PrefetchIterator(train_iter, bufsize=prefetch)
        val_iter = PrefetchIterator(val_iter, bufsize=prefetch)

    return train_iter, val_iter
