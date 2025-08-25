# scripts/build_embeddings.py
from __future__ import annotations

import argparse
import json
import os
import sqlite3
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm

# Local imports
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))  # add repo root to path

from src.embeddings import Embedder, EmbedderConfig, combine_title_text, ensure_unit_norm  # noqa: E402


def read_jsonl(path: Path, limit: int | None = None) -> List[Dict]:
    items = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if line.strip():
                try:
                    items.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON on line {i+1}: {e}") from e
            if limit and len(items) >= limit:
                break
    return items


def ensure_tables(conn: sqlite3.Connection):
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS article_embeddings (
            article_id   INTEGER PRIMARY KEY,
            idx          INTEGER NOT NULL,
            d            INTEGER NOT NULL,
            npy_path     TEXT    NOT NULL,
            model_name   TEXT    NOT NULL,
            normalized   INTEGER NOT NULL,
            created_at   TEXT    NOT NULL
        )
        """
    )
    # Optional: a minimal articles table if you end up writing article rows later.
    conn.commit()


def upsert_article_embeddings(
    conn: sqlite3.Connection,
    rows: List[Tuple[int, int, int, str, str, int, str]],
):
    conn.executemany(
        """
        INSERT INTO article_embeddings(article_id, idx, d, npy_path, model_name, normalized, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(article_id) DO UPDATE SET
          idx=excluded.idx,
          d=excluded.d,
          npy_path=excluded.npy_path,
          model_name=excluded.model_name,
          normalized=excluded.normalized,
          created_at=excluded.created_at
        """,
        rows,
    )
    conn.commit()


def build(args):
    in_path = Path(args.input)
    out_npy = Path(args.out_npy)
    out_db = Path(args.out_db)
    out_npy.parent.mkdir(parents=True, exist_ok=True)
    out_db.parent.mkdir(parents=True, exist_ok=True)

    # 1) Load items
    print(f"[build_embeddings] reading: {in_path}")
    items = read_jsonl(in_path, limit=args.limit)
    if not items:
        raise SystemExit("No items found in input JSONL.")
    print(f"[build_embeddings] loaded {len(items)} items")

    # Resolve id/title/text fields (you can adjust field names with CLI args)
    def get_field(d: Dict, key: str, default=None):
        return d.get(key, default)

    ids: List[int] = []
    texts: List[str] = []

    for it in items:
        art_id = get_field(it, args.id_field)
        if art_id is None:
            # allow fallback to index, but strongly prefer explicit id
            art_id = len(ids)
        title = get_field(it, args.title_field, "")
        body = get_field(it, args.text_field, "")

        ids.append(int(art_id))
        texts.append(
            combine_title_text(
                title=title,
                text=body,
                max_chars=args.max_chars,
                title_weight=args.title_weight,
            )
        )

    # 2) Encode
    cfg = EmbedderConfig(
        model_name=args.model_name,
        normalize=args.normalize,
        batch_size=args.batch_size,
        show_progress=True,
    )
    embedder = Embedder(cfg)
    print(f"[build_embeddings] encoding with {cfg.model_name} (normalize={cfg.normalize})")
    # chunk in manageable batches to reduce peak memory, but expose a progress bar
    embs_batches: List[np.ndarray] = []
    for i in tqdm(range(0, len(texts), args.batch_size), desc="encoding"):
        batch = texts[i : i + args.batch_size]
        vecs = embedder.encode(batch)
        embs_batches.append(vecs)
    embs = np.vstack(embs_batches).astype(np.float32)

    if args.force_post_normalize:
        # Mostly redundant if normalize=True above, but useful for safety
        embs = ensure_unit_norm(embs)

    # 3) Consistency checks
    if len(embs) != len(ids):
        raise AssertionError("Mismatch between embeddings and ids count.")
    if not np.isfinite(embs).all():
        raise AssertionError("Found non-finite values in embeddings.")
    d = embs.shape[1]
    print(f"[build_embeddings] shape: {embs.shape}  (N={len(ids)}, D={d})")

    # 4) Save .npy
    np.save(out_npy, embs)
    print(f"[build_embeddings] wrote: {out_npy}")

    # 5) Write mapping rows to SQLite
    created_at = datetime.now(timezone.utc).isoformat()
    rows = []
    for idx, art_id in enumerate(ids):
        rows.append(
            (
                int(art_id),              # article_id
                int(idx),                 # idx into the npy array
                int(d),                   # embedding dim
                str(out_npy),             # npy_path
                str(cfg.model_name),      # model_name
                1 if bool(cfg.normalize or args.force_post_normalize) else 0,  # normalized
                created_at,               # created_at
            )
        )

    conn = sqlite3.connect(out_db)
    try:
        ensure_tables(conn)
        upsert_article_embeddings(conn, rows)
        print(f"[build_embeddings] wrote {len(rows)} rows to {out_db} (article_embeddings)")
    finally:
        conn.close()

    # 6) Manifest (tiny JSON alongside .npy for reproducibility)
    manifest = {
        "input_jsonl": str(in_path),
        "n_items": len(ids),
        "dim": d,
        "normalize": bool(cfg.normalize or args.force_post_normalize),
        "model_name": cfg.model_name,
        "created_at": created_at,
        "npy_path": str(out_npy),
        "db_path": str(out_db),
        "max_chars": args.max_chars,
        "title_weight": args.title_weight,
    }
    manifest_path = out_npy.with_suffix(".manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"[build_embeddings] wrote manifest: {manifest_path}")
    print("[build_embeddings] DONE ✅")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build article embeddings → .npy + sqlite mapping")
    p.add_argument("--in", dest="input", required=True, help="Processed JSONL (e.g., data/processed/preprocessed_YYYY-MM-DD.jsonl)")
    p.add_argument("--out-npy", default="data/processed/embeddings.npy", help="Path to write embeddings .npy")
    p.add_argument("--out-db", default="data/processed/news.sqlite", help="SQLite DB to record mapping rows")
    p.add_argument("--model-name", default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-chars", type=int, default=1200, help="Truncate article text to this many chars before embedding")
    p.add_argument("--title-weight", type=int, default=1, help="Repeat title N times to upweight topical similarity")
    p.add_argument("--id-field", default="id")
    p.add_argument("--title-field", default="title")
    p.add_argument("--text-field", default="text")
    p.add_argument("--limit", type=int, default=None, help="Optional cap on number of items for quick runs")
    p.add_argument("--normalize", action="store_true", default=True, help="Normalize embeddings at encode time (default True)")
    p.add_argument("--no-normalize", dest="normalize", action="store_false")
    p.add_argument("--force-post-normalize", action="store_true", help="Force L2 normalization after encode as a safety net")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build(args)
