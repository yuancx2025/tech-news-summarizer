"""
Scrape technology news with newspaper3k, store as JSONL + SQLite.
- Discovery: newspaper.build(domain) pulls article URLs from a homepage/section.
- Extraction: Article(url).download() + .parse() (+ optional .nlp()).
- Quality: skip very-short text; dedupe by content hash; UNIQUE(url) at DB level.
- Storage: append JSONL (raw) and upsert into SQLite (processed).
- Config: domains & limits in config/feeds.yml.

Usage:
    python scripts/scrape_newspaper.py \
        --config config/feeds.yml \
        --out-jsonl data/raw/articles.jsonl \
        --out-sqlite data/processed/news.sqlite

Quick tests:
    python scripts/scrape_newspaper.py --domain https://techcrunch.com --limit 10 --no-sqlite --out-jsonl -
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sqlite3
import sys
import time
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional
import json
import yaml
from newspaper import Article, build

# ----------------------------- Utility helpers ----------------------------- #

def sha256(s: Optional[str]) -> str:
    """Stable fingerprint for deduplication based on article text."""
    return hashlib.sha256((s or "").encode("utf-8")).hexdigest()

# def utc_now_iso() -> str:
#     """UTC timestamp in ISO 8601 (for fetched_at)."""
#     return datetime.now(timezone.utc).isoformat()

# def to_iso_or_none(dt) -> Optional[str]:
#     """Safely convert a datetime (or None) to ISO string."""
#     try:
#         return dt.isoformat() if dt else None
#     except Exception:
#         return None

# ----------------------------- Article shaping ----------------------------- #

def article_to_row(a: Article, url: str) -> Dict:
    """
    Convert a parsed newspaper3k Article into a flat dict that works
    for both JSONL and SQLite storage. Lists are JSON-encoded for SQLite.
    """
    authors = a.authors or []
    keywords = getattr(a, "keywords", []) or []

    row = {
        "url": url,
        "title": a.title or None,
        "authors": json.dumps(authors, ensure_ascii=False),
        # "published": to_iso_or_none(getattr(a, "publish_date", None)),
        "text": a.text or "",
        "summary": getattr(a, "summary", None),
        "keywords": json.dumps(keywords, ensure_ascii=False),
        "content_hash": sha256(a.text),
        # "fetched_at": utc_now_iso(),
    }
    return row

def parse_url(url: str, lang: str = "en", request_timeout: int = 20) -> Dict:
    """
    Download + parse a single URL. Calls .nlp() for summary/keywords,
    but won't crash the run if it fails.
    """
    art = Article(url, language=lang, fetch_images=False, request_timeout=request_timeout)
    art.download()
    art.parse()
    try:
        art.nlp()   # optional; adds .summary and .keywords
    except Exception:
        pass
    return article_to_row(art, url)

def crawl_domain(domain: str, limit: int, lang: str, per_request_sleep: float = 0.4) -> List[Dict]:
    """
    Discover up to `limit` article URLs from a domain and parse each.
    Applies basic quality gates and in-run deduplication by content_hash.
    """
    paper = build(domain, memoize_articles=False, fetch_images=False)
    rows: List[Dict] = []
    seen_hashes: set[str] = set()

    for art in paper.articles[: max(0, limit)]:
        try:
            row = parse_url(art.url, lang=lang)
            # Quality gate: skip tiny/boilerplate pages
            if len(row["text"]) < 300:
                continue
            # In-run duplicate: if same content appears under multiple URLs
            if row["content_hash"] in seen_hashes:
                continue
            seen_hashes.add(row["content_hash"])
            rows.append(row)
            # Politeness: tiny delay per fetch to avoid hammering sites
            if per_request_sleep > 0:
                time.sleep(per_request_sleep)
        except Exception as e:
            print(f"[WARN] {art.url} -> {e}", file=sys.stderr)
    return rows

# ----------------------------- Storage: SQLite ----------------------------- #

SCHEMA = """
CREATE TABLE IF NOT EXISTS articles (
  id INTEGER PRIMARY KEY,
  url TEXT UNIQUE,
  title TEXT,
  authors TEXT,
  -- published TEXT,
  text TEXT,
  summary TEXT,
  keywords TEXT,
  content_hash TEXT,
  -- fetched_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_published ON articles(published);
CREATE INDEX IF NOT EXISTS idx_content_hash ON articles(content_hash);
"""

def ensure_db(path: str) -> sqlite3.Connection:
    """Create the SQLite DB (and schema) if missing; return a live connection."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    con = sqlite3.connect(path)
    # Enable WAL for better concurrent read behavior
    try:
        con.execute("PRAGMA journal_mode=WAL;")
    except Exception:
        pass
    with con:
        for statement in SCHEMA.strip().split(";\n"):
            if statement.strip():
                con.execute(statement)
    return con

def upsert_rows(con: sqlite3.Connection, rows: Iterable[Dict]) -> int:
    """
    Insert rows; skip if URL already exists (UNIQUE). Returns number inserted.
    We keep INSERT OR IGNORE to avoid exceptions on duplicates.
    """
    sql = """
    INSERT OR IGNORE INTO articles
      (url, title, authors, published, text, summary, keywords, content_hash, fetched_at)
    VALUES (:url, :title, :authors, :published, :text, :summary, :keywords, :content_hash, :fetched_at)
    """
    with con:
        cur = con.executemany(sql, rows)
        # sqlite3.Cursor.rowcount returns number of rows modified by the *last* statement,
        # but for executemany it generally returns total changes for this batch.
        return cur.rowcount if cur.rowcount is not None else 0

# ----------------------------- Storage: JSONL ------------------------------ #

def write_jsonl(path: str, rows: Iterable[Dict]) -> None:
    """
    Append rows as JSON Lines. If path == "-", write to stdout (handy for piping).
    """
    if path == "-":
        for r in rows:
            print(json.dumps(r, ensure_ascii=False))
        return

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

# ----------------------------- Config loading ----------------------------- #

def load_config(cfg_path: str) -> dict:
    """
    Load YAML config with structure:
    sources:
    - domain: https://techcrunch.com
        limit: 5
        language: en
    - domain: https://thenextweb.com
        limit: 3
        language: en
    """
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

# ----------------------------- CLI / Orchestration ------------------------- #

def main() -> None:
    ap = argparse.ArgumentParser(description="Scrape tech news with newspaper3k")
    
    # Primary mode: read a list of domains from YAML
    ap.add_argument("--config", default="config/feeds.yml",
                    help="YAML file listing sources/domains/limits")
    
    # Convenience overrides (optional): scrape a single domain or a single URL
    ap.add_argument("--domain", help="Override: scrape this domain only (e.g., https://techcrunch.com)")
    ap.add_argument("--url", help="Override: scrape a single article URL")

    # Outputs
    ap.add_argument("--out-jsonl", default="data/raw/articles.jsonl",
                    help="Append JSON Lines here (use '-' for stdout)")
    ap.add_argument("--out-sqlite", default="data/processed/news.sqlite",
                    help="SQLite file path (created if missing)")
    ap.add_argument("--no-sqlite", action="store_true",
                    help="Skip writing to SQLite")

    # Tunables
    ap.add_argument("--limit", type=int, help="If set with --domain, max articles to crawl")
    ap.add_argument("--lang", default="en", help="If set with --domain/--url, language code")
    ap.add_argument("--sleep", type=float, default=0.4, help="Per-request politeness delay (seconds)")

    args = ap.parse_args()

    all_rows: List[Dict] = []

    # --- Override: Single URL mode (quick test / debugging) ---
    if args.url:
        print(f"[INFO] Parsing single URL: {args.url}", file=sys.stderr)
        try:
            row = parse_url(args.url, lang=args.lang)
            if len(row["text"]) >= 300:
                all_rows.append(row)
            else:
                print("[INFO] Skipped (short text).", file=sys.stderr)
        except Exception as e:
            print(f"[ERR] Failed to parse URL: {e}", file=sys.stderr)

    # --- Override: Single domain mode (no YAML needed) ---
    elif args.domain:
        limit = args.limit if args.limit is not None else 30
        print(f"[INFO] Crawling domain: {args.domain} (limit={limit}, lang={args.lang})", file=sys.stderr)
        rows = crawl_domain(args.domain, limit=limit, lang=args.lang, per_request_sleep=args.sleep)
        print(f"[INFO]   Collected {len(rows)} rows", file=sys.stderr)
        all_rows.extend(rows)

    # --- Default: YAML-driven multi-domain crawl ---
    else:
        cfg = load_config(args.config)
        sources = cfg.get("sources", [])
        if not sources:
            print(f"[ERR] No sources found in {args.config}. "
                  f"Provide --domain or populate the YAML.", file=sys.stderr)
            sys.exit(2)

        for src in sources:
            domain = src["domain"]
            limit = int(src.get("limit", 30))
            lang = src.get("language", "en")
            print(f"[INFO] Crawling {domain} (limit={limit}, lang={lang})", file=sys.stderr)
            rows = crawl_domain(domain, limit=limit, lang=lang, per_request_sleep=args.sleep)
            print(f"[INFO]   Collected {len(rows)} rows", file=sys.stderr)
            all_rows.extend(rows)

    # If nothing valid was scraped, exit gracefully
    if not all_rows:
        print("[INFO] No rows scraped.", file=sys.stderr)
        return

    # Always write JSONL (raw append or stdout)
    write_jsonl(args.out_jsonl, all_rows)
    # print(f"[INFO] Wrote JSONL → {args.out-jsonl if args.out_jsonl == '-' else args.out_jsonl}", file=sys.stderr)
    dest = "stdout" if args.out_jsonl == "-" else args.out_jsonl
    print(f"[INFO] Wrote JSONL → {dest}", file=sys.stderr)

    # Optionally write SQLite (processed / dedup-friendly)
    if not args.no_sqlite:
        con = ensure_db(args.out_sqlite)
        inserted = upsert_rows(con, all_rows)
        con.close()
        print(f"[INFO] Upserted {inserted} new rows into {args.out_sqlite}", file=sys.stderr)


if __name__ == "__main__":
    main()
