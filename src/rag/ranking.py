# src/rag/ranking.py
"""Scoring, ranking, and deduplication utilities for retrieved documents."""
from __future__ import annotations
from typing import List, Any, Dict, Iterable, Tuple
from datetime import datetime, timezone
import numpy as np
import re

# --------- time & recency ----------
def _parse_iso8601(s: str) -> datetime:
    # Expect "YYYY-MM-DDTHH:MM:SSZ" (UTC). Fall back to naive now if missing.
    if not s:
        return datetime.now(timezone.utc)
    try:
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        return datetime.fromisoformat(s).astimezone(timezone.utc)
    except Exception:
        return datetime.now(timezone.utc)

def age_days(iso_utc: str) -> float:
    dt = _parse_iso8601(iso_utc)
    return max(0.0, (datetime.now(timezone.utc) - dt).total_seconds() / 86400.0)

def recency_decay(age_days_val: float, half_life_days: float = 7.0) -> float:
    # Exponential half-life decay in [0,1]
    if half_life_days <= 0:
        return 1.0
    return float(np.exp(-age_days_val * np.log(2) / half_life_days))

# --------- vector scoring ----------
def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)

def fetch_vectors_for_ids(vs, ids: List[str]) -> Dict[str, np.ndarray]:
    # Use Chroma's underlying collection to fetch embeddings
    res = vs._collection.get(ids=ids, include=["embeddings"])  # type: ignore[attr-defined]
    out: Dict[str, np.ndarray] = {}
    for _id, emb in zip(res.get("ids", []), res.get("embeddings", [])):
        out[_id] = np.array(emb, dtype=float)
    return out

def score_and_rank_candidates(
    *,
    query: str,
    docs: List[Any],
    vs,
    emb, 
    sim_weight: float = 0.65,
    recency_weight: float = 0.35,
    recency_half_life_days: float = 7.0,
) -> List[Any]:
    # Embed query once
    qv = np.array(emb.embed_query(query), dtype=float)

    # Get vectors for candidate doc IDs
    ids = [d.metadata["doc_id"] for d in docs]
    vecs = fetch_vectors_for_ids(vs, ids)

    scored: List[Tuple[float, Any]] = []
    for d in docs:
        did = d.metadata["doc_id"]
        sim = cosine_sim(qv, vecs.get(did, np.zeros_like(qv)))
        rdec = recency_decay(age_days(d.metadata.get("published_at", "")), recency_half_life_days)
        score = sim_weight * sim + recency_weight * rdec
        scored.append((score, d))

    # Sort by score desc
    scored.sort(key=lambda x: x[0], reverse=True)
    ranked = [d for _, d in scored]
    return ranked

# --------- deduping: titles (press-release near-dups) ----------
_token_re = re.compile(r"[^\w]+", re.UNICODE)

def _tokenize(s: str) -> set:
    return {t for t in _token_re.split(s.lower()) if t and not t.isdigit()}

def jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    a, b = set(a), set(b)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)

def filter_near_duplicate_titles(items: List[Any], threshold: float = 0.85) -> List[Any]:
    kept: List[Any] = []
    seen_tokens: List[set] = []
    for d in items:
        title = d.metadata.get("title") or ""
        toks = _tokenize(title)
        dup = any(jaccard(toks, t2) >= threshold for t2 in seen_tokens)
        if dup:
            continue
        kept.append(d)
        seen_tokens.append(toks)
    return kept
