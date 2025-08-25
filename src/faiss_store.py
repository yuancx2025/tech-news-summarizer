from __future__ import annotations
import faiss
import numpy as np
from pathlib import Path
from typing import Tuple, Optional

def load_embeddings_npy(npy_path: str | Path, memmap: bool = True) -> np.ndarray:
    """
    Load embeddings as float32. Use memmap to avoid loading all into RAM (nice for CLIs).
    For the web app, you typically won't need this (you'll query FAISS only).
    """
    npy_path = str(npy_path)
    if memmap:
        mm = np.load(npy_path, mmap_mode="r")
        # Ensure float32 view
        return mm.astype(np.float32)
    return np.load(npy_path).astype(np.float32)

# --------- Flat (exact) index: great up to ~1–2M vectors on a decent box ---------

def build_index_flat(embs: np.ndarray) -> faiss.Index:
    """
    Expect L2-normalized vectors. Uses inner-product to emulate cosine similarity.
    """
    d = int(embs.shape[1])
    index = faiss.IndexFlatIP(d)
    index.add(embs.astype(np.float32))
    return index

# --------- IVF+PQ (compressed ANN): for large corpora or low RAM ---------

def build_index_ivfpq(
    embs: np.ndarray,
    nlist: int = 4096,
    m: int = 16,
    nbits: int = 8,
    train_size: int = 200_000,
    seed: int = 123,
) -> faiss.Index:
    """
    Inverted File w/ Product Quantization:
      - nlist: #coarse clusters (more = better recall, slower build)
      - m, nbits: PQ code size (m sub-vectors * nbits each)
    Assumes normalized vectors and embs.shape = (N, d)
    """
    rs = faiss.StandardGpuResources() if faiss.get_num_gpus() > 0 else None
    d = int(embs.shape[1])

    quantizer = faiss.IndexFlatIP(d)
    index = faiss.IndexIVFPQ(quantizer, d, nlist, m, nbits, faiss.METRIC_INNER_PRODUCT)

    # Train on a subset (or all if small)
    np.random.seed(seed)
    N = len(embs)
    train_idx = np.random.choice(N, size=min(train_size, N), replace=False)
    index.train(embs[train_idx].astype(np.float32))
    index.add(embs.astype(np.float32))

    # Speed/recall tradeoff at query time; set a sensible default
    index.nprobe = min(64, nlist)  # search this many lists
    return index

# --------- I/O and search ---------

def write_index(index: faiss.Index, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(path))

def read_index(path: str | Path) -> faiss.Index:
    return faiss.read_index(str(path))

def search(
    index: faiss.Index, query_vec: np.ndarray, topk: int = 50
) -> Tuple[np.ndarray, np.ndarray]:
    """
    query_vec: shape (d,) or (1, d), must be float32 and (ideally) unit-normalized
    Returns (scores, indices)
    """
    q = query_vec.astype(np.float32)
    if q.ndim == 1:
        q = q[None, :]
    D, I = index.search(q, topk)
    return D[0], I[0]

def batch_search(
    index: faiss.Index, query_mat: np.ndarray, topk: int = 50
) -> Tuple[np.ndarray, np.ndarray]:
    """
    query_mat: shape (B, d)
    Returns (scores[B, topk], indices[B, topk])
    """
    Q = query_mat.astype(np.float32)
    return index.search(Q, topk)
