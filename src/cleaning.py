# src/cleaning.py
from __future__ import annotations
import hashlib, re
from datetime import datetime, timezone
from typing import Dict, List, Optional
from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode
from unidecode import unidecode

MIN_CHARS = 300

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
def validate_row(row: Dict, min_chars: int = MIN_CHARS_DEFAULT) -> List[str]:
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
