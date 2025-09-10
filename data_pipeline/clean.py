# data_pipeline/clean.py
"""
Unified clean + export + preprocess pipeline

Inputs: raw JSONL from `data_pipeline.scrape` (or any schema-compatible file)
Outputs:
  - Clean CSV + JSONL
  - Preprocessed JSONL (+ manifest)
  - Optional RAG chunks JSONL

Typical usage:
  python -m data_pipeline.clean \
    --in-jsonl data/raw/articles.jsonl \
    --out-csv data/processed/articles_clean.csv \
    --out-jsonl data/processed/articles_clean.jsonl \
    --out-pre-jsonl data/processed/preprocessed_$(date +%F).jsonl \
    --tokenizer-model facebook/bart-large-cnn --truncate-to 1000 \
    --sentencer nltk --min-chars 300 --min-words 80 \
    --make-chunks --chunk-tokens 800 --chunk-overlap 120
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import statistics
import subprocess
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional

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
            # Apply SSL fix before any NLTK operations
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
            print(f"Warning: NLTK sentence splitting failed ({e}), using fallback method")
            sentences = re.split(r"[.!?]+", text)
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
        words = (text or "").split()
        i, step = 0, max_len - overlap
        if step <= 0:
            step = max_len
        while i < len(words):
            yield " ".join(words[i : i + max_len])
            i += max(1, step)
        return
    ids = _TOK(text, truncation=False)["input_ids"]
    start, step = 0, max_len - overlap
    if step <= 0:
        step = max_len
    while start < len(ids):
        piece = ids[start : start + max_len]
        yield _TOK.decode(piece, skip_special_tokens=True)
        start += max(1, step)

# ------------------- Load raw JSONL -------------------

def _infer_source(u: str) -> str:
    try:
        from urllib.parse import urlparse
        return urlparse(u).netloc.lower().replace("www.", "") or "unknown"
    except Exception:
        return "unknown"

def load_raw_jsonl(path: str) -> pd.DataFrame:
    """Read raw JSONL (each line a dict). Returns a DataFrame."""
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                records.append(obj)
            except Exception:
                # skip malformed line
                continue
    if not records:
        return pd.DataFrame(columns=["url", "title", "published", "text"])

    df = pd.DataFrame(records)

    # Normalize shapes
    if "published" not in df.columns and "date" in df.columns:
        df = df.rename(columns={"date": "published"})
    if "text" not in df.columns and "content" in df.columns:
        df = df.rename(columns={"content": "text"})

    # ensure columns exist
    for col in ["url", "title", "published", "text", "content_hash", "fetched_at", "summary"]:
        if col not in df.columns:
            df[col] = ""

    # authors/keywords may come as list or string
    for col in ["authors", "keywords"]:
        if col not in df.columns:
            df[col] = []
    return df

# ------------------- Main -------------------

def main():
    ap = argparse.ArgumentParser(
        description="Clean+export+preprocess from raw JSONL to clean CSV/JSONL and preprocessed JSONL"
    )

    # Input
    ap.add_argument("--in-jsonl", required=True, help="Path to raw JSONL file")

    # Filters
    ap.add_argument("--date-from", default=None, help="inclusive (YYYY-MM-DD)")
    ap.add_argument("--date-to", default=None, help="inclusive (YYYY-MM-DD)")
    ap.add_argument("--sources", nargs="*", default=None, help="limit to these sources (domain names)")
    ap.add_argument("--min-chars", type=int, default=300, help="drop if full text shorter than this many chars")
    ap.add_argument("--min-words", type=int, default=80, help="drop if fewer words after normalization")

    # Clean exports
    ap.add_argument("--out-csv", required=True, help="clean CSV output")
    ap.add_argument("--out-jsonl", required=True, help="clean JSONL output")

    # Preprocess
    ap.add_argument("--out-pre-jsonl", required=True, help="preprocessed JSONL with model_input, sentences, token stats")
    ap.add_argument("--manifest", default=None, help="manifest JSON path (auto from out-pre-jsonl if omitted)")
    ap.add_argument("--sentencer", choices=["nltk", "spacy"], default="nltk")
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

    for p in [args.out_csv, args.out_jsonl, args.out_pre_jsonl, chunks_path]:
        parent = pathlib.Path(p).parent
        if "{date}" in str(p):  # ignore template path until used
            continue
        parent.mkdir(parents=True, exist_ok=True)
    if args.make_chunks:
        pathlib.Path(chunks_path).parent.mkdir(parents=True, exist_ok=True)

    # ---------- 1) LOAD ----------
    df = load_raw_jsonl(args.in_jsonl)
    if args.limit > 0:
        df = df.head(args.limit).copy()

    # minimal normalization & quality
    df["title"] = df["title"].astype(str).str.strip()
    df["url"] = df["url"].astype(str).str.strip()
    df["text"] = df["text"].astype(str).map(normalize_whitespace)

    if "source" not in df.columns:
        df["source"] = df["url"].map(_infer_source)
    else:
        df["source"] = df["source"].astype(str).str.strip().replace("", "unknown")

    # date filters (attempt to parse; drop unparseable)
    def _safe_iso(v: Any) -> Optional[str]:
        try:
            return iso8601(str(v))
        except Exception:
            return None

    df["published_iso"] = df["published"].apply(_safe_iso)
    df = df[df["published_iso"].notna()].copy()

    if args.date_from:
        df = df[df["published_iso"] >= f"{args.date_from}T00:00:00Z"].copy()
    if args.date_to:
        df = df[df["published_iso"] <= f"{args.date_to}T23:59:59Z"].copy()
    if args.sources:
        allow = {s.lower() for s in args.sources}
        df = df[df["source"].str.lower().isin(allow)].copy()

    # drop malformed / too short
    df["chars"] = df["text"].map(len)
    df = df[(df["title"] != "") & (df["url"] != "") & (df["chars"] >= args.min_chars)].copy()

    # dedupe: prefer content_hash, else URL
    key_cols = [c for c in ["content_hash", "url"] if c in df.columns]
    df = df.sort_values("published_iso").drop_duplicates(subset=key_cols or ["url"], keep="last")

    # authors/keywords stringify if list-like
    def _stringify_listish(val) -> str:
        if val is None:
            return ""
        if isinstance(val, str):
            v = val.strip()
            # try parse JSON array-in-string
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

    for col in ["authors", "keywords"]:
        if col in df.columns:
            df[col] = df[col].apply(_stringify_listish)
        else:
            df[col] = ""

    # ---------- 2) EXPORT CLEAN ----------
    export_cols = [
        "url",
        "title",
        "published_iso",
        "source",
        "text",
        "summary",
        "authors",
        "keywords",
        "sentiment",
        "fetched_at",
        "content_hash",
    ]
    export_cols = [c for c in export_cols if c in df.columns]
    df_out = df[export_cols].rename(columns={"published_iso": "published"}).copy()
    df_out.to_csv(args.out_csv, index=False, encoding="utf-8")
    with open(args.out_jsonl, "w", encoding="utf-8") as f:
        for _, row in df_out.iterrows():
            f.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")
    print(f"[OK] Clean CSV: {len(df_out)} rows → {args.out_csv}")
    print(f"[OK] Clean JSONL: {len(df_out)} rows → {args.out_jsonl}")

    # ---------- 3) PREPROCESS ----------
    df_proc = df_out.copy()
    df_proc["text"] = df_proc["text"].astype(str)
    df_proc["words"] = df_proc["text"].map(lambda t: len((t or "").split()))
    before = len(df_proc)
    df_proc = df_proc[df_proc["words"] >= args.min_words].copy()
    dropped_short = before - len(df_proc)

    # sentence split
    df_proc["sentences"] = df_proc["text"].map(lambda t: sent_split(t, mode=args.sentencer))

    # tokenizer & truncation
    load_tokenizer(args.tokenizer_model)
    token_stats: List[int] = []
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
                "title": row.get("title", ""),
                "url": row.get("url", ""),
                "source": row.get("source", ""),
                "date": row.get("published", ""),
                "full_text": text,
                "model_input": model_input,
                "tokens_full": int(tlen) if tlen is not None else None,
                "tokens_input": int(tokens_input) if tokens_input is not None else None,
                "sentences": row.get("sentences", []),
                "summary": row.get("summary", ""),
                "authors": row.get("authors", ""),
                "keywords": row.get("keywords", ""),
                "content_hash": row.get("content_hash", ""),
                "fetched_at": row.get("fetched_at", ""),
            }
            outp.write(json.dumps(rec, ensure_ascii=False) + "\n")
            kept += 1

            if args.make_chunks:
                if "{date}" in args.chunks_out:
                    chunks_path = args.chunks_out.format(date=datetime.utcnow().strftime("%Y-%m-%d"))
                else:
                    chunks_path = args.chunks_out
                with open(chunks_path, "a", encoding="utf-8") as ch:
                    idx = 0
                    for piece in chunkify_by_tokens(text, args.chunk_tokens, args.chunk_overlap):
                        ch.write(
                            json.dumps(
                                {
                                    "parent_url": rec["url"],
                                    "chunk_id": f"{rec['url']}#{idx}",
                                    "chunk_index": idx,
                                    "chunk_text": piece,
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                        idx += 1
                        chunks_written += 1

    # ---------- 4) MANIFEST ----------
    manifest: Dict[str, Any] = {
        "input_jsonl": args.in_jsonl,
        "filters": {
            "date_from": args.date_from,
            "date_to": args.date_to,
            "sources": args.sources,
            "min_chars": args.min_chars,
            "min_words": args.min_words,
        },
        "outputs": {
            "clean_csv": args.out_csv,
            "clean_jsonl": args.out_jsonl,
            "pre_jsonl": args.out_pre_jsonl,
            "chunks_out": (args.chunks_out if args.make_chunks else None),
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
            k = max(0, min(len(token_stats_sorted) - 1, int(round((p / 100.0) * (len(token_stats_sorted) - 1)))))
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
        print(f"[OK] Chunks: {args.chunks_out} (+{chunks_written} pieces)")

if __name__ == "__main__":
    main()
