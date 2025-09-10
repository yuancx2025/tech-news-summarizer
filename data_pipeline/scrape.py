# data_pipeline/scrape.py
"""
Scrape technology/finance news with newspaper3k and append to JSONL.

- Discovery: newspaper.build(domain) pulls article URLs from a homepage/section.
- Extraction: Article(url).download() + .parse() (+ optional .nlp()).
- Quality: skip very-short text; in-run dedupe by content_hash.
- Storage: append JSONL (raw). No SQLite.

Config: domains & limits in config/feeds.yml.

Usage:
    python -m data_pipeline.scrape \
        --config config/feeds.yml \
        --out-jsonl data/raw/articles.jsonl

Quick tests:
    python -m data_pipeline.scrape \
        --domain https://techcrunch.com \
        --limit 10 \
        --out-jsonl -
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time
from typing import Dict, Iterable, List, Optional

import hashlib, re
from datetime import datetime, timezone
from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode
from unidecode import unidecode

# allow `from src.cleaning import ...` etc.
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

import yaml
from newspaper import Article, build

# from src.cleaning import content_hash, utc_now_iso, validate_row, MIN_CHARS
MIN_CHARS = 300

# ----------------------------- Utility helpers ----------------------------- #

def to_iso_or_none(dt) -> Optional[str]:
    """Safely convert a datetime (or None) to ISO string."""
    try:
        return dt.isoformat() if dt else None
    except Exception:
        return None

# ---------- URL normalization ----------
def normalize_url(u: str) -> str:
    """Strip tracking params (?utm_*, fbclid, gclid, ref, etc.) and fragments."""
    try:
        scheme, netloc, path, query, _ = urlsplit(u)
        keep = []
        for k, v in parse_qsl(query, keep_blank_values=True):
            lk = k.lower()
            if lk.startswith("utm_") or lk in {"fbclid", "gclid", "mc_cid", "mc_eid", "ref"}:
                continue
            keep.append((k, v))
        query2 = urlencode(keep, doseq=True)
        return urlunsplit((scheme, netloc, path, query2, ""))  # drop fragment
    except Exception:
        return u

# ---------- Text cleanup ----------
_ws_collapse = re.compile(r"[ \t]+")
_nl_collapse = re.compile(r"\n{3,}")

def clean_text(t: Optional[str]) -> str:
    """Whitespace normalization + Unicode folding for stable hashing & nicer evals."""
    if not t:
        return ""
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    t = _nl_collapse.sub("\n\n", t)      # keep max 2 newlines
    t = _ws_collapse.sub(" ", t)
    t = t.strip()
    t = unidecode(t)                     # “smart quotes” -> straight quotes, etc.
    return t

# ---------- Hashing & timestamps ----------
def content_hash(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

# ---------- Validation ----------
def validate_row(row: Dict, min_chars: int = MIN_CHARS) -> List[str]:
    """
    Return list of problems for a row (empty list means OK).
    Checks: title present, text length.
    """
    problems = []
    if not (row.get("title") or "").strip():
        problems.append("title")
    txt = row.get("text") or ""
    if len(txt) < min_chars:
        problems.append(f"text<{min_chars}")
    return problems


# ----------------------------- Article shaping ----------------------------- #

def article_to_row(a: Article, url: str) -> Dict:
    """
    Convert a parsed newspaper3k Article into a flat dict for JSONL storage.
    Lists are kept as JSON-serializable Python lists.
    """
    authors = a.authors or []
    keywords = getattr(a, "keywords", []) or []

    return {
        "url": url,
        "title": a.title or None,
        "authors": authors,
        "published": to_iso_or_none(getattr(a, "publish_date", None)),
        "text": a.text or "",
        "summary": getattr(a, "summary", None),
        "keywords": keywords,
        "content_hash": content_hash(a.text),
        "sentiment": "pending",
        "fetched_at": utc_now_iso(),
    }


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

            # Validate required fields
            issues = validate_row(row, MIN_CHARS)
            if issues:
                print(f"[WARN] Skipping {row['url']} -> {','.join(issues)}", file=sys.stderr)
                continue

            # In-run dedupe
            if row["content_hash"] in seen_hashes:
                print(f"[INFO] Duplicate by content_hash, skipping {row['url']}", file=sys.stderr)
                continue
            seen_hashes.add(row["content_hash"])

            rows.append(row)
            if per_request_sleep > 0:
                time.sleep(per_request_sleep)

        except Exception as e:
            print(f"[WARN] {art.url} -> {e}", file=sys.stderr)
    return rows


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
    ap = argparse.ArgumentParser(description="Scrape tech/finance news with newspaper3k")

    # Primary mode: read a list of domains from YAML
    ap.add_argument("--config", default="config/feeds.yml",
                    help="YAML file listing sources/domains/limits")

    # Convenience overrides (optional): scrape a single domain or a single URL
    ap.add_argument("--domain", help="Override: scrape this domain only (e.g., https://techcrunch.com)")
    ap.add_argument("--url", help="Override: scrape a single article URL")

    # Output
    ap.add_argument("--out-jsonl", default="data/raw/articles.jsonl",
                    help="Append JSON Lines here (use '-' for stdout)")

    # Tunables
    ap.add_argument("--limit", type=int, help="If set with --domain, max articles to crawl")
    ap.add_argument("--lang", default="en", help="If set with --domain/--url, language code")
    ap.add_argument("--sleep", type=float, default=0.4, help="Per-request politeness delay (seconds)")

    args = ap.parse_args()

    all_rows: List[Dict] = []

    # --- Single URL mode ---
    if args.url:
        print(f"[INFO] Parsing single URL: {args.url}", file=sys.stderr)
        try:
            row = parse_url(args.url, lang=args.lang)
            if len(row["text"]) >= MIN_CHARS:
                all_rows.append(row)
            else:
                print("[INFO] Skipped (short text).", file=sys.stderr)
        except Exception as e:
            print(f"[ERR] Failed to parse URL: {e}", file=sys.stderr)

    # --- Single domain mode ---
    elif args.domain:
        limit = args.limit if args.limit is not None else 30
        print(f"[INFO] Crawling domain: {args.domain} (limit={limit}, lang={args.lang})", file=sys.stderr)
        rows = crawl_domain(args.domain, limit=limit, lang=args.lang, per_request_sleep=args.sleep)
        print(f"[INFO]   Collected {len(rows)} rows", file=sys.stderr)
        all_rows.extend(rows)

    # --- YAML-driven multi-domain crawl ---
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
    dest = "stdout" if args.out_jsonl == "-" else args.out_jsonl
    print(f"[INFO] Wrote JSONL → {dest}", file=sys.stderr)


if __name__ == "__main__":
    main()
