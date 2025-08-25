from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))  # repo root

from src.indexes.faiss_store import (
    load_embeddings_npy,
    build_index_flat,
    build_index_ivfpq,
    write_index,
)

def parse_args():
    p = argparse.ArgumentParser(description="Build FAISS index from embeddings.npy")
    p.add_argument("--npy", required=True, help="Path to embeddings.npy")
    p.add_argument("--out", default="data/processed/news.faiss", help="Output index path")
    p.add_argument("--type", choices=["flat", "ivfpq"], default="flat", help="Index type")
    p.add_argument("--nlist", type=int, default=4096, help="IVF lists (ivfpq)")
    p.add_argument("--m", type=int, default=16, help="PQ m (ivfpq)")
    p.add_argument("--nbits", type=int, default=8, help="PQ nbits per sub-vector (ivfpq)")
    p.add_argument("--train-size", type=int, default=200_000, help="Train subset size (ivfpq)")
    p.add_argument("--no-memmap", action="store_true", help="Disable memmap when loading .npy")
    return p.parse_args()

def main():
    args = parse_args()
    embs = load_embeddings_npy(args.npy, memmap=not args.no_memmap)
    print(f"[faiss] embeddings: shape={embs.shape} dtype={embs.dtype} (memmap={not args.no_memmap})")

    if args.type == "flat":
        index = build_index_flat(embs)
        print("[faiss] built FLAT (exact) index")
    else:
        index = build_index_ivfpq(
            embs,
            nlist=args.nlist,
            m=args.m,
            nbits=args.nbits,
            train_size=args.train_size,
        )
        print(f"[faiss] built IVF+PQ index (nlist={args.nlist}, m={args.m}, nbits={args.nbits})")

    write_index(index, args.out)
    print(f"[faiss] wrote index to: {args.out}")
    print("[faiss] DONE ✅")

if __name__ == "__main__":
    main()
