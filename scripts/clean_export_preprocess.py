#!/usr/bin/env python3
"""
Unified clean + export + preprocess pipeline.

This script merges the main functionality of:
- clean_and_export.py  → quality filtering from SQLite and export to CSV/JSONL
- preprocess_from_clean.py → sentence splitting, tokenizer-aware truncation,
                              optional chunking, and manifest

Goal: from ONE command, produce clean CSV/JSONL (+ optional clean SQLite table)
and a model-ready preprocessed JSONL (+ manifest, optional chunks).

Typical usage:
  python scripts/clean_export_preprocess.py \
    --in-sqlite data/processed/news.sqlite --table articles \
    --out-csv data/processed/articles_clean.csv \
    --out-jsonl data/processed/articles_clean.jsonl \
    --out-pre-jsonl data/processed/preprocessed_$(date +%F).jsonl \
    --tokenizer-model facebook/bart-large-cnn --truncate-to 1000 \
    --sentencer nltk --min-chars 300 --min-words 80 \
    --make-chunks --chunk-tokens 800 --chunk-overlap 120

Notes:
- Cleaning and exporting happen first (min chars, non-empty title, date/source filters,
  dedup by content_hash/url), written to CSV + JSONL, and optionally to a clean SQLite table.
- Preprocessing then operates on the cleaned rows in-memory (no re-read) to add:
  sentences, tokens_full, model_input (truncated), tokens_input. It writes a preprocessed JSONL
  and a manifest with token stats and drop counts. Optional chunk file supports RAG in Week 3.

"""

from __future__ import annotations
import argparse, json, os, re, sqlite3, statistics, subprocess, sys, pathlib
from datetime import datetime, timezone
from typing import Optional, Iterable, Dict, Any, List

import pandas as pd
from tqdm import tqdm

# ------------------- Tokenizer (lazy) -------------------
_TOK = None

def load_tokenizer(model_name: Optional[str]):
    global _TOK
    if model_name:
        from transformers import AutoTokenizer
        _TOK = AutoTokenizer.from_pretrained(model_name)
    return _TOK

def token_len(text: str) -> Optional[int]:
    if _TOK is None:
        return None
    return len(_TOK(text, truncation=False)["input_ids"])

def truncate(text: str, max_tokens: int) -> str:
    enc = _TOK(text, max_length=max_tokens, truncation=True)
    return _TOK.decode(enc["input_ids"], skip_special_tokens=True)

# ------------------- Utilities -------------------
def iso8601(dt_str: str) -> str:
    """Normalize to UTC ISO-8601 with 'Z' suffix."""
    from dateutil import parser as dateparser
    dt = dateparser.parse(dt_str) if dt_str else None
    if dt is None:
        raise ValueError("unparseable date")
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")

def sent_split(text: str, mode: str = "nltk") -> List[str]:
    if mode == "spacy":
        import spacy
        nlp = spacy.load("en_core_web_sm")
        return [s.text.strip() for s in nlp(text).sents if s.strip()]
    else:
        try:
            import nltk
            # Try to download punkt with SSL verification disabled if needed
            try:
                nltk.download("punkt", quiet=True)
            except Exception:
                # Fallback: download without SSL verification
                import ssl
                try:
                    _create_unverified_https_context = ssl._create_unverified_context
                except AttributeError:
                    pass
                else:
                    ssl._create_default_https_context = _create_unverified_https_context
                nltk.download("punkt", quiet=True)
            
            from nltk.tokenize import sent_tokenize
            return [s.strip() for s in sent_tokenize(text) if s.strip()]
        except Exception as e:
            # Fallback to simple sentence splitting if NLTK fails
            print(f"Warning: NLTK sentence splitting failed ({e}), using fallback method")
            # Simple sentence splitting by common sentence endings
            import re
            sentences = re.split(r'[.!?]+', text)
            return [s.strip() for s in sentences if s.strip()]

def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()

def get_git_commit() -> Optional[str]:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None

def chunkify_by_tokens(text: str, max_len: int, overlap: int) -> Iterable[str]:
    """Tokenizer-based overlapping chunks; word-based fallback if tokenizer not loaded."""
    if _TOK is None:
        # fallback by words
        words = (text or "").split()
        i, step = 0, max_len - overlap
        if step <= 0: step = max_len  # avoid infinite loop
        while i < len(words):
            yield " ".join(words[i:i+max_len])
            i += max(1, step)
        return
    ids = _TOK(text, truncation=False)["input_ids"]
    start, step = 0, max_len - overlap
    if step <= 0: step = max_len
    while start < len(ids):
        piece = ids[start:start+max_len]
        yield _TOK.decode(piece, skip_special_tokens=True)
        start += max(1, step)

# ------------------- DB I/O -------------------
def fetch_from_sqlite(db_path: str, table: str, date_from: Optional[str], date_to: Optional[str],
                      sources: Optional[List[str]], min_chars: int) -> pd.DataFrame:
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    cur = con.cursor()

    # Build WHERE filters
    where = []
    params = []
    if date_from:
        where.append("published >= ?")
        params.append(date_from)
    if date_to:
        where.append("published <= ?")
        params.append(date_to)
    if sources:
        placeholders = ",".join("?" for _ in sources)
        where.append(f"source IN ({placeholders})")
        params.extend(sources)

    sql = f"SELECT * FROM {table}"
    if where:
        sql += " WHERE " + " AND ".join(where)

    rows = [dict(r) for r in cur.execute(sql, params)]
    con.close()

    if not rows:
        return pd.DataFrame(columns=["id","url","title","published","text","source"])

    df = pd.DataFrame(rows)

    # Minimal normalization to align to expected columns
    # try to infer/rename common fields if needed
    rename_map = {}
    if "date" in df.columns and "published" not in df.columns:
        rename_map["date"] = "published"
    if "content" in df.columns and "text" not in df.columns:
        rename_map["content"] = "text"
    if rename_map:
        df = df.rename(columns=rename_map)

    # Quality gates similar to clean_and_export
    # ensure required columns exist
    for col in ["id","url","title","published","text"]:
        if col not in df.columns:
            df[col] = ""

    df["title"] = df["title"].astype(str).str.strip()
    df["url"] = df["url"].astype(str).str.strip()
    df["text"] = df["text"].astype(str).map(normalize_whitespace)
    if "source" in df.columns:
        df["source"] = df["source"].astype(str).str.strip()
    else:
        # infer from url
        from urllib.parse import urlparse
        def infer_src(u): 
            try:
                netloc = urlparse(u).netloc.lower().replace("www.","")
                return netloc or "unknown"
            except Exception:
                return "unknown"
        df["source"] = df["url"].map(infer_src)

    # drop malformed / too short
    df["chars"] = df["text"].map(len)
    df = df[(df["title"]!="") & (df["url"]!="") & (df["chars"]>=min_chars)].copy()

    # dedupe: prefer content_hash, else URL, else text hash
    key_cols = [c for c in ["content_hash","url"] if c in df.columns]
    if "content_hash" in key_cols:
        df = df.sort_values("published").drop_duplicates(subset="content_hash", keep="last")
    else:
        df = df.sort_values("published").drop_duplicates(subset=key_cols or ["url"], keep="last")

    # flatten authors/keywords if present as JSON-like
    for col in ["authors","authors_str","keywords","keywords_str"]:
        if col in df.columns:
            df[col] = df[col].apply(_stringify_listish)

    # Ensure published ISO 8601
    df["published"] = df["published"].apply(lambda x: _safe_iso(x))
    df = df[df["published"].notna()].copy()

    return df

def _stringify_listish(val) -> str:
    # Accept JSON text, Python list repr, or already string
    if val is None:
        return ""
    if isinstance(val, str):
        v = val.strip()
        if v.startswith("[") and v.endswith("]"):
            try:
                arr = json.loads(v)
                if isinstance(arr, list):
                    return ", ".join([str(a) for a in arr])
            except Exception:
                return v
        return v
    if isinstance(val, (list, tuple)):
        return ", ".join([str(a) for a in val])
    return str(val)

def _safe_iso(v: Any) -> Optional[str]:
    try:
        return iso8601(str(v))
    except Exception:
        return None

# ------------------- Main -------------------
def main():
    ap = argparse.ArgumentParser(description="Unified clean+export+preprocess from SQLite to CSV/JSONL and preprocessed JSONL")
    # Input
    ap.add_argument("--in-sqlite", required=True, help="Path to SQLite DB (from scraper)")
    ap.add_argument("--table", default="articles")
    ap.add_argument("--date-from", default=None, help="inclusive (YYYY-MM-DD)")
    ap.add_argument("--date-to", default=None, help="inclusive (YYYY-MM-DD)")
    ap.add_argument("--sources", nargs="*", default=None, help="limit to these sources (domain names)")
    ap.add_argument("--min-chars", type=int, default=300, help="drop if full text shorter than this many chars")

    # Clean exports
    ap.add_argument("--out-csv", required=True, help="clean CSV output")
    ap.add_argument("--out-jsonl", required=True, help="clean JSONL output")
    ap.add_argument("--out-clean-sqlite", default=None, help="optional: write cleaned rows to a new SQLite DB at this path (table clean_articles)")

    # Preprocess
    ap.add_argument("--out-pre-jsonl", required=True, help="preprocessed JSONL with model_input, sentences, token stats")
    ap.add_argument("--manifest", default=None, help="manifest JSON path (auto from out-pre-jsonl if omitted)")
    ap.add_argument("--sentencer", choices=["nltk","spacy"], default="nltk")
    ap.add_argument("--min-words", type=int, default=80, help="drop if fewer words after normalization (preprocess stage)")
    ap.add_argument("--tokenizer-model", default="facebook/bart-large-cnn")
    ap.add_argument("--truncate-to", type=int, default=1000)

    # Chunking (optional for RAG)
    ap.add_argument("--make-chunks", action="store_true")
    ap.add_argument("--chunk-tokens", type=int, default=800)
    ap.add_argument("--chunk-overlap", type=int, default=120)
    ap.add_argument("--chunks-out", default="data/chunks/chunks_{date}.jsonl")

    ap.add_argument("--limit", type=int, default=0, help="limit rows for quick dry run")

    args = ap.parse_args()

    stamp = datetime.utcnow().strftime("%Y-%m-%d")
    manifest_path = args.manifest or args.out_pre_jsonl.replace(".jsonl", "_manifest.json")
    chunks_path = args.chunks_out.format(date=stamp)

    pathlib.Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.out_jsonl).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.out_pre_jsonl).parent.mkdir(parents=True, exist_ok=True)
    if args.make_chunks:
        pathlib.Path(chunks_path).parent.mkdir(parents=True, exist_ok=True)
    if args.out_clean_sqlite:
        pathlib.Path(args.out_clean_sqlite).parent.mkdir(parents=True, exist_ok=True)

    # ---------- 1) CLEAN from SQLite ----------
    df = fetch_from_sqlite(
        db_path=args.in_sqlite,
        table=args.table,
        date_from=args.date_from,
        date_to=args.date_to,
        sources=args.sources,
        min_chars=args.min_chars,
    )

    if args.limit > 0:
        df = df.head(args.limit).copy()

    # write clean CSV
    export_cols = [
        "id","url","title","published","source","text",
        "summary","authors","authors_str","keywords","keywords_str",
        "sentiment","fetched_at","content_hash"
    ]
    # Keep only columns that exist
    export_cols = [c for c in export_cols if c in df.columns]
    df_out = df[export_cols].copy()
    df_out.to_csv(args.out_csv, index=False, encoding="utf-8")
    print(f"[OK] Clean CSV: {len(df_out)} rows → {args.out_csv}")

    # write clean JSONL
    with open(args.out_jsonl, "w", encoding="utf-8") as f:
        for _, row in df_out.iterrows():
            f.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")
    print(f"[OK] Clean JSONL: {len(df_out)} rows → {args.out_jsonl}")

    # optionally write clean SQLite
    if args.out_clean_sqlite:
        con = sqlite3.connect(args.out_clean_sqlite)
        df_out.to_sql("clean_articles", con, if_exists="replace", index=False)
        con.close()
        print(f"[OK] Clean SQLite: table clean_articles → {args.out_clean_sqlite}")

    # ---------- 2) PREPROCESS in-memory ----------
    # Apply min-words and sentence splitting
    df_proc = df_out.copy()
    df_proc["text"] = df_proc["text"].astype(str)
    df_proc["words"] = df_proc["text"].map(lambda t: len((t or '').split()))
    before = len(df_proc)
    df_proc = df_proc[df_proc["words"] >= args.min_words].copy()
    dropped_short = before - len(df_proc)

    # Ensure ISO date field for downstream
    df_proc["date"] = df_proc["published"].apply(_safe_iso)
    df_proc = df_proc[df_proc["date"].notna()].copy()

    # sentence split
    df_proc["sentences"] = df_proc["text"].map(lambda t: sent_split(t, mode=args.sentencer))

    # tokenizer & truncation
    load_tokenizer(args.tokenizer_model)
    token_stats = []
    kept = 0
    chunks_written = 0

    with open(args.out_pre_jsonl, "w", encoding="utf-8") as outp:
        for _, row in tqdm(df_proc.iterrows(), total=len(df_proc), desc="Preprocessing"):
            text = row["text"]
            if _TOK is not None:
                tlen = token_len(text)
                token_stats.append(tlen)
                model_input = truncate(text, args.truncate_to)
                tokens_input = token_len(model_input)
            else:
                tlen = None
                model_input = " ".join(text.split()[: args.truncate_to])
                tokens_input = None

            rec = {
                "id": str(row.get("id","")),
                "title": row.get("title",""),
                "url": row.get("url",""),
                "source": row.get("source",""),
                "date": row.get("date",""),
                "full_text": text,
                "model_input": model_input,
                "tokens_full": int(tlen) if tlen is not None else None,
                "tokens_input": int(tokens_input) if tokens_input is not None else None,
                "sentences": row.get("sentences", []),
                "summary": row.get("summary",""),
                "authors": row.get("authors","") or row.get("authors_str",""),
                "keywords": row.get("keywords","") or row.get("keywords_str",""),
                "content_hash": row.get("content_hash",""),
                "fetched_at": row.get("fetched_at",""),
            }
            outp.write(json.dumps(rec, ensure_ascii=False) + "\n")
            kept += 1

            if args.make_chunks:
                with open(chunks_path, "a", encoding="utf-8") as ch:
                    idx = 0
                    for piece in chunkify_by_tokens(text, args.chunk_tokens, args.chunk_overlap):
                        ch.write(json.dumps({
                            "parent_id": rec["id"],
                            "chunk_id": f"{rec['id']}:{idx}",
                            "chunk_index": idx,
                            "chunk_text": piece,
                        }, ensure_ascii=False) + "\n")
                        idx += 1
                        chunks_written += 1

    # manifest
    manifest: Dict[str, Any] = {
        "input_sqlite": args.in_sqlite,
        "table": args.table,
        "filters": {
            "date_from": args.date_from,
            "date_to": args.date_to,
            "sources": args.sources,
            "min_chars": args.min_chars,
        },
        "outputs": {
            "clean_csv": args.out_csv,
            "clean_jsonl": args.out_jsonl,
            "clean_sqlite": args.out_clean_sqlite,
            "pre_jsonl": args.out_pre_jsonl,
            "chunks_out": (chunks_path if args.make_chunks else None),
        },
        "created_at": datetime.utcnow().isoformat() + "Z",
        "git_commit": get_git_commit(),
        "rows_clean": int(len(df_out)),
        "rows_pre_kept": int(kept),
        "pre_dropped_short": int(dropped_short),
        "tokenizer": args.tokenizer_model,
        "truncate_to_tokens": int(args.truncate_to),
        "sentencer": args.sentencer,
        "chunks_written": int(chunks_written),
    }

    if token_stats:
        token_stats_sorted = sorted(token_stats)
        def pct(p):
            k = max(0, min(len(token_stats_sorted)-1, int(round((p/100.0)*(len(token_stats_sorted)-1)))))
            return token_stats_sorted[k]
        manifest["token_stats"] = {
            "counted": len(token_stats),
            "mean": float(statistics.fmean(token_stats)),
            "p50": float(pct(50)),
            "p90": float(pct(90)),
            "p95": float(pct(95)),
            "p99": float(pct(99)),
            "over_1024_pct": float(sum(t > 1024 for t in token_stats)) / len(token_stats),
        }

    pathlib.Path(manifest_path).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[OK] Preprocessed JSONL: {kept} rows → {args.out_pre_jsonl}")
    print(f"[OK] Manifest: {manifest_path}")
    if args.make_chunks:
        print(f"[OK] Chunks: {chunks_path} ({chunks_written} pieces)")

if __name__ == "__main__":
    main()
