import numpy as np
from pathlib import Path

from src.faiss_store import build_index_flat, write_index, read_index, search, batch_search


def test_faiss_build_search(tmp_path):
    # 5 unit-normalized 3D points on axes/diagonals
    X = np.array([
        [1,0,0],
        [0,1,0],
        [0,0,1],
        [1,1,0],
        [0,1,1],
    ], dtype=np.float32)
    X = X / np.linalg.norm(X, axis=1, keepdims=True)

    index = build_index_flat(X)

    # Query near [1,0,0]
    q = np.array([1, 0, 0], dtype=np.float32)
    scores, idxs = search(index, q, topk=3)
    assert idxs[0] == 0  # nearest is itself
    assert scores[0] == max(scores)  # best score first

    # Batch search
    Q = np.stack([q, np.array([0, 1, 0], dtype=np.float32)], axis=0)
    Ds, Is = batch_search(index, Q, topk=2)
    assert Is.shape == (2, 2)
    assert Is[0, 0] == 0 and Is[1, 0] in (1, 3, 4)

    # IO round-trip
    out = tmp_path / "news.faiss"
    write_index(index, out)
    index2 = read_index(out)
    s2, i2 = search(index2, q, topk=1)
    assert i2[0] == idxs[0]
