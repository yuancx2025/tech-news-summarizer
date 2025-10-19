"""Utilities to extract normalized article catalog data from Chroma."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import pandas as pd
import yaml
from langchain_chroma import Chroma

DEFAULT_PAGE_SIZE = 200


@dataclass
class ArticleRecord:
    """In-memory representation of a deduplicated article."""

    article_id: str
    title: Optional[str]
    url: Optional[str]
    source: Optional[str]
    published_at: Optional[datetime]
    tickers: List[str] = field(default_factory=list)
    sectors: List[str] = field(default_factory=list)
    text: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


def _iter_collection(
    collection: Any,
    *,
    where: Optional[Mapping[str, Any]] = None,
    page_size: int = DEFAULT_PAGE_SIZE,
    include: Optional[Sequence[str]] = None,
) -> Iterator[Tuple[str, str, Dict[str, Any]]]:
    """Yield documents from a Chroma collection in pages."""

    include_fields = list(include or ["metadatas", "documents"])
    offset = 0
    while True:
        res = collection.get(  # type: ignore[attr-defined]
            where=dict(where or {}),
            limit=page_size,
            offset=offset,
            include=["ids", *include_fields],
        )
        ids: List[str] = res.get("ids", [])
        documents: List[str] = res.get("documents", [])
        metadatas: List[Dict[str, Any]] = res.get("metadatas", [])
        if not ids:
            break

        for doc_id, content, meta in zip(ids, documents, metadatas):
            yield doc_id, content or "", dict(meta or {})

        if len(ids) < page_size:
            break
        offset += len(ids)


_TICKER_REGEX = re.compile(r"\b([A-Z]{1,5})(?:\.[A-Z]{1,2})?\b")


def detect_tickers_from_text(text: str, *, allowed: Optional[Sequence[str]] = None) -> List[str]:
    """Detect potential ticker symbols from free text using a conservative regex."""

    if not text:
        return []
    candidates = {match.group(1) for match in _TICKER_REGEX.finditer(text)}
    if allowed is not None:
        allowed_set = {a.upper() for a in allowed}
        candidates = {c for c in candidates if c in allowed_set}
    return sorted(candidates)


def _parse_datetime(value: Any) -> Optional[datetime]:
    if isinstance(value, datetime):
        return value
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(value)
        except (ValueError, OSError):
            return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        # Normalize trailing Z for UTC
        text = text.replace("Z", "+00:00") if text.endswith("Z") else text
        try:
            return datetime.fromisoformat(text)
        except ValueError:
            return None
    return None


def _coerce_str_list(value: Any, upper: bool = False) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        items = [v.strip() for v in value.split(",") if v and v.strip()]
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        items = [str(v).strip() for v in value if str(v).strip()]
    else:
        return []
    if upper:
        items = [item.upper() for item in items]
    return sorted(dict.fromkeys(items))


def normalize_metadata(meta: Mapping[str, Any]) -> Dict[str, Any]:
    """Normalize per-chunk metadata into a consistent schema."""

    normalized: Dict[str, Any] = dict(meta)
    normalized["tickers"] = _coerce_str_list(meta.get("tickers"), upper=True)
    normalized["sectors"] = _coerce_str_list(meta.get("sectors"), upper=False)
    normalized["published_at"] = _parse_datetime(meta.get("published_at"))
    return normalized


def load_ticker_lookup(path: str | Path) -> Dict[str, Dict[str, Any]]:
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Ticker lookup not found at {cfg_path}")
    data = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    if isinstance(data, dict) and "tickers" in data:
        data = data["tickers"]
    lookup: Dict[str, Dict[str, Any]] = {}
    for key, value in data.items():
        if not isinstance(value, Mapping):
            lookup[str(key).upper()] = {"name": str(value), "sector": None}
            continue
        entry = {k: v for k, v in value.items()}
        entry.setdefault("sector", None)
        lookup[str(key).upper()] = entry
    return lookup


def _aggregate_articles(
    docs: Iterable[Tuple[str, str, Dict[str, Any]]]
) -> Dict[str, ArticleRecord]:
    articles: Dict[str, ArticleRecord] = {}
    for _doc_id, content, meta in docs:
        normalized = normalize_metadata(meta)
        article_id = normalized.get("article_id") or normalized.get("doc_id")
        if not article_id and _doc_id:
            article_id = _doc_id.split("::")[0]
        if not article_id:
            continue
        record = articles.setdefault(
            article_id,
            ArticleRecord(
                article_id=article_id,
                title=normalized.get("title"),
                url=normalized.get("url_canonical") or normalized.get("url"),
                source=normalized.get("source_short") or normalized.get("source"),
                published_at=normalized.get("published_at"),
                tickers=[],
                sectors=[],
                text="",
                metadata={},
            ),
        )

        if normalized.get("published_at") and (not record.published_at or normalized["published_at"] > record.published_at):
            record.published_at = normalized["published_at"]
        if normalized.get("title") and not record.title:
            record.title = normalized["title"]
        if normalized.get("url_canonical") and not record.url:
            record.url = normalized["url_canonical"]
        if normalized.get("source_short") and not record.source:
            record.source = normalized["source_short"]

        if normalized.get("tickers"):
            merged = sorted({*record.tickers, *normalized["tickers"]})
            record.tickers = merged
        if normalized.get("sectors"):
            merged = sorted({*record.sectors, *normalized["sectors"]})
            record.sectors = merged
        record.text = (record.text + "\n\n" + content).strip() if record.text else content
        record.metadata.update(normalized)
    return articles


def _enrich_articles(
    articles: MutableMapping[str, ArticleRecord],
    ticker_lookup: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> None:
    allowed = list(ticker_lookup.keys()) if ticker_lookup else None
    for record in articles.values():
        detected = (
            detect_tickers_from_text(
                " ".join(filter(None, [record.title or "", record.text])), allowed=allowed
            )
            if allowed
            else []
        )
        combined = sorted({*record.tickers, *detected})
        record.tickers = combined
        if ticker_lookup:
            sector_set = {sec for sec in record.sectors if sec}
            for ticker in record.tickers:
                info = ticker_lookup.get(ticker)
                sector = (info or {}).get("sector")
                if sector:
                    sector_set.add(sector)
            record.sectors = sorted(sector_set)
        record.metadata["tickers"] = record.tickers
        record.metadata["sectors"] = record.sectors
        if record.published_at:
            record.metadata["published_at"] = record.published_at.isoformat()


def collect_articles(
    rag_config_path: str | Path,
    ticker_config_path: Optional[str | Path] = None,
    *,
    page_size: int = DEFAULT_PAGE_SIZE,
    where: Optional[Mapping[str, Any]] = None,
) -> pd.DataFrame:
    """Extract a deduplicated article catalog from Chroma."""

    cfg_path = Path(rag_config_path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"RAG config not found at {cfg_path}")
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    chroma_dir = cfg.get("chroma_dir")
    if not chroma_dir:
        raise KeyError("chroma_dir missing from RAG config")

    vs = Chroma(persist_directory=chroma_dir, embedding_function=None)
    docs = _iter_collection(vs._collection, where=where, page_size=page_size)
    aggregated = _aggregate_articles(docs)

    ticker_lookup = load_ticker_lookup(ticker_config_path) if ticker_config_path else None
    _enrich_articles(aggregated, ticker_lookup)

    rows = []
    for record in aggregated.values():
        rows.append(
            {
                "article_id": record.article_id,
                "title": record.title,
                "url": record.url,
                "source": record.source,
                "published_at": record.published_at,
                "tickers": record.tickers,
                "sectors": record.sectors,
                "text": record.text,
                "metadata": record.metadata,
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty and df["published_at"].notnull().any():
        df["published_at"] = pd.to_datetime(df["published_at"], utc=True)
    return df
