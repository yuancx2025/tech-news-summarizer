# src/rag/retriever.py
from __future__ import annotations
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Dict, Any, Tuple

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

# -------- utilities --------
def _utc_floor(days_back: int) -> str:
    dt = datetime.now(timezone.utc) - timedelta(days=days_back)
    return dt.strftime("%Y-%m-%dT00:00:00Z")

def build_where(
    *,
    days_back: int,
    language: str = "en",
    tickers: Optional[List[str]] = None,
    sources: Optional[List[str]] = None,
) -> Dict[str, Any]:
    where: Dict[str, Any] = {
        "published_at": {"$gte": _utc_floor(days_back)},
        "language": language,
    }
    if tickers:
        where["tickers"] = {"$in": [t.upper() for t in tickers]}
    if sources:
        where["source_short"] = {"$in": sources}
    return where

def _dedup_by_article(docs: List[Any], k_final: int) -> List[Any]:
    seen = set()
    out = []
    for d in docs:
        aid = d.metadata.get("article_id") or d.metadata.get("doc_id", "").split("::")[0]
        if aid in seen:
            continue
        seen.add(aid)
        out.append(d)
        if len(out) >= k_final:
            break
    return out


# -------- main retrieval entrypoint --------
def retrieve_chroma_mmr(
    *,
    chroma_dir: str,
    embedding_model: str = "text-embedding-3-small",
    query: str,
    k_final: int = 8,
    fetch_k: int = 60,
    lambda_mult: float = 0.6,
    days_back_start: int = 60,
    days_back_max: int = 365,
    language: str = "en",
    tickers: Optional[List[str]] = None,
    sources: Optional[List[str]] = None,
) -> List[Any]:
    """
    Retrieval with:
    - Chroma metadata filters (date/ticker/source/language)
    - search_type='mmr' for diversity
    - auto-expanding recency window if sparse results
    - light dedup by article_id
    Returns a list of LC Documents (chunks) with metadata attached.
    """
    emb = OpenAIEmbeddings(model=embedding_model)
    vs = Chroma(persist_directory=chroma_dir, embedding_function=emb)

    days = days_back_start
    docs: List[Any] = []

    while days <= days_back_max:
        where = build_where(days_back=days, language=language, tickers=tickers, sources=sources)
        retriever = vs.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": k_final,
                "fetch_k": fetch_k,
                "lambda_mult": lambda_mult,
                "where": where,
            },
        )
        docs = retriever.invoke(query)

        if len(docs) >= k_final or days >= days_back_max:
            break

        # Auto-expand window (double, capped at max)
        days = min(days * 2, days_back_max)

    # Keep only one chunk per article by default
    return _dedup_by_article(docs, k_final)


# -------- convenience wrapper using rag.yml --------
def retrieve_with_config(query: str, cfg: Dict[str, Any], *, tickers=None, sources=None) -> List[Any]:
    r = cfg["retrieval"]
    return retrieve_chroma_mmr(
        chroma_dir=cfg["chroma_dir"],
        embedding_model=cfg["embedding_model"],
        query=query,
        k_final=r["k_final"],
        fetch_k=r["fetch_k"],
        lambda_mult=r["lambda_mult"],
        days_back_start=r["days_back_start"],
        days_back_max=r["days_back_max"],
        language=r.get("language", "en"),
        tickers=tickers,
        sources=sources,
    )
