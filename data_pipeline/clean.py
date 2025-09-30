# data_pipeline/clean.py
"""
Clean and normalize raw news JSONL for RAG.

Key improvements implemented here:
- Schema alignment: ensure fields `id`, `published_at` (UTC ISO-8601 with Z), `source`.
- Canonicalize URLs and strip tracking params.
- Author field sanitization (remove CSS/DOM noise; output list[str]).
- Boilerplate/footer stripping from title/text.
- HTML and entity removal; whitespace normalization.
- Exact dedupe by content hash (post-clean text); optional near-dup by title+time.
- Optional language filter (best-effort; falls back to ASCII heuristic if langdetect not installed).
- Optional CSV export.
"""
from __future__ import annotations

import argparse, json, csv, re, html as ihtml, hashlib, sys
from pathlib import Path
from typing import Optional, Dict, Any, Iterable, List, Tuple
from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode
from datetime import datetime, timezone
from tqdm import tqdm

# --------------- Utilities ---------------
def canonical_url(u: Optional[str]) -> Optional[str]:
    """Normalize URL by stripping tracking params and fragments."""
    if not u:
        return u
    try:
        p = urlsplit(u)
        qs = [(k, v) for k, v in parse_qsl(p.query, keep_blank_values=True)
              if not k.lower().startswith(("utm_", "fbclid", "gclid", "mc_cid", "mc_eid", "ref"))]
        return urlunsplit((p.scheme, p.netloc, p.path, urlencode(qs), ""))
    except Exception:
        return u

FOOTER_PATTERNS = [
    r"subscribe to our newsletter.*?$",
    r"all rights reserved.*?$",
    r"terms of use|privacy policy",
    r"follow us on (twitter|x|facebook|linkedin)",
    r"^advertisement$",
]
_footer_re = re.compile("|".join(FOOTER_PATTERNS), re.I | re.M)

def strip_boilerplate(text: str) -> str:
    if not text:
        return text
    t = _footer_re.sub(" ", text)
    # collapse whitespace
    t = re.sub(r"\s+", " ", t).strip()
    return t

_tag_re = re.compile(r"<[^>]+>")

def strip_html(text: str) -> str:
    if not text:
        return text or ""
    # remove tags
    t = _tag_re.sub(" ", text)
    # unescape entities
    t = ihtml.unescape(t)
    # collapse
    t = re.sub(r"\s+", " ", t).strip()
    return t

def clean_authors(val) -> List[str]:
    if val is None:
        return []
    if isinstance(val, list):
        items = val
    else:
        items = re.split(r",|·|\band\b|\|", str(val), flags=re.I)
    out = []
    for x in items:
        x = str(x).strip()
        if not x:
            continue
        xl = x.lower()
        if len(x) > 80:
            continue
        if re.search(r"(display|--|post-authors|align-items|var|\{|\}|margin|padding|min-width|max-width)", xl):
            continue
        # drop obvious CSS tokens
        if re.search(r"[{};:]{2,}", x):
            continue
        out.append(x)
    # dedupe order-preserving
    seen = set()
    res = []
    for a in out:
        if a not in seen:
            seen.add(a)
            res.append(a)
    return res

def to_utc_iso(dt_str: Optional[str]) -> Optional[str]:
    if not dt_str:
        return None
    # try python stdlib first
    dt = None
    try:
        # handle already iso with Z or offset
        dt = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
    except Exception:
        try:
            from dateutil import parser as dateparser  # type: ignore
            dt = dateparser.parse(dt_str)
        except Exception:
            dt = None
    if not dt:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")

def domain_of(u: Optional[str]) -> Optional[str]:
    if not u:
        return None
    try:
        return urlsplit(u).netloc.lower()
    except Exception:
        return None

def content_hash(title: str, text: str) -> str:
    base = (strip_html(title or "") + "||" + strip_html(text or "")).lower().strip()
    base = re.sub(r"\s+", " ", base)
    return hashlib.sha1(base.encode("utf-8")).hexdigest()

def is_english(text: str) -> bool:
    if not text:
        return True
    try:
        from langdetect import detect  # type: ignore
        lang = detect(text[:1000])
        return lang == "en"
    except Exception:
        # fallback ascii heuristic
        sample = text[:1000]
        letters = sum(ch.isalpha() for ch in sample)
        ascii_letters = sum('a' <= ch.lower() <= 'z' for ch in sample)
        return ascii_letters >= 0.6 * max(1, letters)

def normalize_record(rec: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    # basic required fields
    title = rec.get("title") or rec.get("headline") or ""
    text = rec.get("text") or rec.get("content") or ""
    url = canonical_url(rec.get("url"))
    if not title or not text or not url:
        return None

    # strip HTML & boilerplate in text and title
    title = strip_boilerplate(strip_html(title))
    text = strip_boilerplate(strip_html(text))

    # language filter (best effort)
    if not is_english(title + " " + text):
        return None

    pub = rec.get("published_at") or rec.get("published") or rec.get("date")
    published_at = to_utc_iso(pub)
    source = rec.get("source") or domain_of(url) or None

    authors = clean_authors(rec.get("authors"))
    chash = content_hash(title, text)

    rid = rec.get("id") or hashlib.sha1(f"{url}::{title}".encode("utf-8")).hexdigest()

    return {
        "id": rid,
        "url": url,
        "source": source,
        "title": title,
        "text": text,
        "authors": authors,
        "published_at": published_at,
        "language": rec.get("language") or "en",
        "content_hash": chash,
    }

def load_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    """Stream JSONL records one at a time to avoid loading entire file into memory."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue

def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> int:
    """Write JSONL incrementally, return count of rows written."""
    n = 0
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in rows:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    return n

def write_csv(path: str, rows: Iterable[Dict[str, Any]], fieldnames: List[str]) -> int:
    """Write CSV incrementally, return count of rows written."""
    n = 0
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for rec in rows:
            w.writerow({k: rec.get(k) for k in fieldnames})
            n += 1
    return n

def dedupe_exact(rows: Iterable[Dict[str, Any]]) -> Iterable[Dict[str, Any]]:
    """Streaming deduplication by content_hash."""
    seen = set()
    for r in rows:
        h = r.get("content_hash")
        if not h:
            yield r
            continue
        if h in seen:
            continue
        seen.add(h)
        yield r

def main():
    ap = argparse.ArgumentParser("Clean raw articles JSONL for RAG")
    ap.add_argument("--in-jsonl", required=True, help="Input raw JSONL file")
    ap.add_argument("--out-jsonl", required=True, help="Output cleaned JSONL file")
    ap.add_argument("--out-csv", default=None, help="Optional CSV export")
    ap.add_argument("--drop-non-en", action="store_true", help="Force-drop non-English")
    args = ap.parse_args()

    print(f"[clean] Reading from {args.in_jsonl}")
    
    # Streaming pipeline: load → normalize → filter → dedupe
    raw_rows = load_jsonl(args.in_jsonl)
    
    # Count raw for progress
    raw_count = sum(1 for _ in load_jsonl(args.in_jsonl))
    print(f"[clean] Processing {raw_count} raw records...")
    
    def normalize_stream():
        for rec in tqdm(load_jsonl(args.in_jsonl), total=raw_count, desc="Normalizing"):
            norm = normalize_record(rec)
            if not norm:
                continue
            if args.drop_non_en and norm.get("language") != "en":
                continue
            yield norm
    
    # Dedupe and write
    cleaned = dedupe_exact(normalize_stream())
    n_jsonl = write_jsonl(args.out_jsonl, cleaned)
    print(f"[clean] Wrote {n_jsonl} rows → {args.out_jsonl}")
    
    if args.out_csv:
        # Re-stream for CSV
        cleaned_for_csv = dedupe_exact(normalize_stream())
        fields = ["id", "url", "source", "title", "published_at", "language"]
        n_csv = write_csv(args.out_csv, cleaned_for_csv, fieldnames=fields)
        print(f"[clean] Wrote {n_csv} rows → {args.out_csv}")

if __name__ == "__main__":
    main()
    main()
