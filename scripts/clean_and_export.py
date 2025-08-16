"""
Clean + export articles from SQLite to a tidy CSV for Week-2.
- Applies quality filters (min chars, non-empty title)
- Optional date & source filters
- De-duplicates by content_hash
- Decodes authors/keywords JSON fields into readable text
Usage examples:
  python scripts/clean_and_export.py --db data/processed/news.sqlite --out data/processed/articles_clean.csv --min-chars 400
  python scripts/clean_and_export.py --db data/processed/news.sqlite --out data/processed/articles_2025-08.csv --date-from 2025-08-01 --date-to 2025-08-31
  python scripts/clean_and_export.py --sources techcrunch.com theverge.com --min-chars 500
"""

from __future__ import annotations
import argparse
import json
import os
import sqlite3
from urllib.parse import urlsplit

import pandas as pd


def build_query(include_sources: list[str] | None) -> tuple[str, list]:
    """
    Build a parameterized SQL query. If include_sources is provided, we add a WHERE ... IN (...)
    on the host extracted in Python (we'll filter hosts after load for simplicity/portability).
    """
    # Base selection: keep only columns we actually need downstream
    sql = """
    SELECT
      id,
      url,
      title,
      published,
      text,
      summary,
      keywords,
      authors,
      content_hash,
      sentiment,
      fetched_at
    FROM articles
    WHERE (title IS NOT NULL AND TRIM(title) <> '')
      AND length(text) >= ?
    """
    params: list = []  # first param will be min_chars; appended later
    # date range filters get appended at runtime (if provided)
    return sql, params


def add_date_filters(sql: str, params: list, date_from: str | None, date_to: str | None) -> tuple[str, list]:
    """
    Append ISO date range filters to the WHERE clause when provided.
    We assume 'published' is ISO-8601 (string compare works for YYYY-MM-DD prefixes).
    """
    if date_from:
        sql += " AND (published IS NOT NULL AND substr(published,1,10) >= ?)"
        params.append(date_from)
    if date_to:
        sql += " AND (published IS NOT NULL AND substr(published,1,10) <= ?)"
        params.append(date_to)
    return sql, params


def source_host(url: str) -> str:
    try:
        host = urlsplit(url).netloc.lower()
        return host[4:] if host.startswith("www.") else host
    except Exception:
        return ""


def decode_json_field(val) -> list[str]:
    """authors/keywords are stored as JSON strings; return a Python list."""
    if val is None or val == "":
        return []
    if isinstance(val, list):
        return val
    try:
        out = json.loads(val)
        return out if isinstance(out, list) else []
    except Exception:
        return []


def main():
    ap = argparse.ArgumentParser(description="Clean + export articles from SQLite to CSV")
    ap.add_argument("--db", default="data/processed/news.sqlite", help="Path to SQLite database")
    ap.add_argument("--out", default="data/processed/articles_clean.csv", help="Output CSV path")
    ap.add_argument("--min-chars", type=int, default=300, help="Minimum article text length")
    ap.add_argument("--date-from", help="YYYY-MM-DD (inclusive)")
    ap.add_argument("--date-to", help="YYYY-MM-DD (inclusive)")
    ap.add_argument("--sources", nargs="*",
                    help="Filter by source host(s), e.g. techcrunch.com theverge.com (match on URL host)")
    ap.add_argument("--limit", type=int, default=0, help="Optional maximum number of rows after filtering (0 = no limit)")
    ap.add_argument("--dedupe", action="store_true", help="Drop duplicate content_hash rows (keep first)")

    args = ap.parse_args()

    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    # Build SQL
    sql, params = build_query(args.sources)
    sql, params = add_date_filters(sql, params, args.date_from, args.date_to)

    # First param is min_chars
    params.insert(0, args.min_chars)

    # Read from SQLite
    con = sqlite3.connect(args.db)
    try:
        df = pd.read_sql_query(sql, con, params=params)
    finally:
        con.close()

    if df.empty:
        print("[INFO] No rows matched the filters; nothing to export.")
        return

    # Add a normalized source host column
    df["source"] = df["url"].map(source_host)

    # Optional filter by sources
    if args.sources:
        wanted = {s.lower() for s in args.sources}
        df = df[df["source"].isin(wanted)]

    # Decode authors/keywords JSON strings -> readable text columns
    df["authors_list"] = df["authors"].map(decode_json_field)
    df["keywords_list"] = df["keywords"].map(decode_json_field)
    # also provide comma-joined strings for quick viewing
    df["authors_str"] = df["authors_list"].map(lambda xs: ", ".join(xs) if xs else "")
    df["keywords_str"] = df["keywords_list"].map(lambda xs: ", ".join(xs) if xs else "")

    # De-duplicate by content and by URL as a fallback
    if args.dedupe:
        before = len(df)
        df = df.sort_values(["published", "fetched_at", "id"], ascending=[False, False, True], na_position="last")
        df = df.drop_duplicates(subset=["content_hash"], keep="first")
        df = df.drop_duplicates(subset=["url"], keep="first")
        after = len(df)
        print(f"[INFO] Deduped rows by content/url: {before} -> {after}")

    # Optional limit after all filtering
    if args.limit and args.limit > 0:
        df = df.head(args.limit)

    # Choose export columns (tidy, human-friendly)
    export_cols = [
        "id", "source", "url", "title", "published",
        "text", "summary", "authors_str", "keywords_str",
        "sentiment", "fetched_at", "content_hash"
    ]
    # Keep only columns that exist
    export_cols = [c for c in export_cols if c in df.columns]
    df = df[export_cols]

    # Write CSV (utf-8)
    df.to_csv(args.out, index=False)
    print(f"[OK] Exported {len(df)} rows → {args.out}")


if __name__ == "__main__":
    main()
