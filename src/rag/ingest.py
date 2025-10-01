# src/rag/ingest.py
from __future__ import annotations
import hashlib, re, orjson
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode
from datetime import datetime, timezone
from dateutil import parser as dateparser
from tqdm import tqdm

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from src.embeddings import get_embeddings


# ---------- helpers ----------
def _normalize_ws(text: str) -> str:
    text = text.replace("\u00A0", " ")
    return re.sub(r"\s+", " ", text).strip()

def _sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()

def _canonicalize_url(url: Optional[str]) -> Optional[str]:
    if not url:
        return None
    try:
        u = urlparse(url)
        # Drop tracking params (utm_*, fbclid, etc.)
        q = [(k, v) for (k, v) in parse_qsl(u.query, keep_blank_values=True) if not k.lower().startswith("utm") and k.lower() != "fbclid"]
        u2 = u._replace(query=urlencode(q))
        # Normalize scheme/host casing
        return urlunparse(u2._replace(scheme=u2.scheme.lower(), netloc=u2.netloc.lower()))
    except Exception:
        return url

def _to_utc_iso(ts: Optional[str]) -> Optional[str]:
    if not ts:
        return None
    try:
        dt = dateparser.parse(ts)
        if not dt.tzinfo:
            dt = dt.replace(tzinfo=timezone.utc)
        dt_utc = dt.astimezone(timezone.utc)
        return dt_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
    except Exception:
        return None

_SENT_SPLIT = re.compile(r"(?<=[\.\?\!])\s+")
def _lead3(text: str) -> str:
    sents = _SENT_SPLIT.split(_normalize_ws(text))
    return " ".join(sents[:3]).strip()

def _load_jsonl(fp: Path) -> Iterable[Dict]:
    with fp.open("rb") as f:
        for line in f:
            if line.strip():
                yield orjson.loads(line)

def _source_short(name: Optional[str]) -> Optional[str]:
    if not name: return None
    # quick-and-dirty shortener
    name = name.strip()
    mapping = {
        "The Wall Street Journal": "WSJ",
        "Financial Times": "FT",
        "The New York Times": "NYT",
        "Bloomberg": "Bloomberg",
        "Reuters": "Reuters",
    }
    return mapping.get(name, name)


# ---------- core: convert article -> chunks with metadata ----------
def make_chunks_from_article(
    a: Dict,
    *,
    chunk_size: int,
    chunk_overlap: int,
    min_chars: int,
    separators: Optional[List[str]] = None,
) -> Tuple[List[Document], Optional[str]]:
    text = _normalize_ws(a.get("text") or a.get("content") or "")
    if len(text) < min_chars:
        return [], None

    title = _normalize_ws(a.get("title", ""))
    url = _canonicalize_url(a.get("url"))
    published_at = _to_utc_iso(a.get("published_at"))
    language = a.get("language") or "en"

    # Identity / lineage
    article_id = a.get("id") or _sha1(f"{url or ''}::{title}")
    content_hash = _sha1(text)

    # Explainability
    lead3 = _lead3(text)
    source = _source_short(a.get("source") or a.get("publisher"))

    # Facets
    tickers = a.get("tickers") or []
    if isinstance(tickers, str):
        # allow comma-separated strings
        tickers = [t.strip().upper() for t in tickers.split(",") if t.strip()]
    else:
        tickers = [str(t).upper() for t in tickers]

    section = a.get("section") or a.get("category")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators or ["\n\n", "\n", ". ", "? ", "! ", " ", ""],
    )
    parts = splitter.split_text(text)

    docs: List[Document] = []
    n = len(parts)
    for i, chunk in enumerate(parts):
        doc_id = f"{article_id}::{i}"
        meta = {
            "article_id": article_id,
            "doc_id": doc_id,
            "url_canonical": url,
            "source_short": source,
            "published_at": published_at,  # UTC ISO-8601
            "content_sha1": content_hash,

            "tickers": tickers,
            "section": section,
            "language": language,

            "title": title,
            "lead3": lead3,

            "chunk_idx": i,
            "n_chunks": n,
        }
        docs.append(Document(page_content=chunk, metadata=meta))
    return docs, content_hash


# ---------- ingestion: idempotent & batched into Chroma ----------
def ingest_articles_to_chroma(
    input_path: str,
    chroma_dir: str,
    embedding_model: str = "models/text-embedding-004",
    chunk_size: int = 1000,
    chunk_overlap: int = 100,
    min_chars: int = 300,
    separators: Optional[List[str]] = None,
    batch_limit: int = 1500,
):
    in_fp = Path(input_path)
    assert in_fp.exists(), f"Input not found: {in_fp}"

    emb = get_embeddings(provider="gemini", model_name=embedding_model)
    vs = Chroma(persist_directory=chroma_dir, embedding_function=emb)

    seen_hashes = set()  # dedup within this run

    def already_indexed(content_sha1: str) -> bool:
        # O(1) if repeated in this run; O(n) micro-query to Chroma otherwise.
        if content_sha1 in seen_hashes:
            return True
        try:
            # Use underlying collection to query metadata directly
            res = vs._collection.get(where={"content_sha1": content_sha1}, limit=1)  # type: ignore[attr-defined]
            return bool(res and res.get("ids"))
        except Exception:
            return False

    buffer: List[Document] = []
    added_chunks = 0
    total_articles = 0
    skipped_articles = 0

    for a in tqdm(_load_jsonl(in_fp), desc="Ingesting"):
        total_articles += 1
        docs, content_hash = make_chunks_from_article(
            a,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            min_chars=min_chars,
            separators=separators,
        )
        if not docs or not content_hash:
            skipped_articles += 1
            continue
        if already_indexed(content_hash):
            skipped_articles += 1
            continue

        buffer.extend(docs)
        seen_hashes.add(content_hash)

        if len(buffer) >= batch_limit:
            _flush_add(vs, buffer)
            added_chunks += len(buffer)
            buffer.clear()

    if buffer:
        _flush_add(vs, buffer)
        added_chunks += len(buffer)

    print(
        f"[DONE] articles_total={total_articles} "
        f"articles_skipped={skipped_articles} "
        f"chunks_added={added_chunks} "
        f"chroma_dir={chroma_dir}"
    )

def _flush_add(vs: Chroma, docs: List[Document]) -> None:
    # Convert complex metadata to ChromaDB-compatible format
    processed_docs = []
    for doc in docs:
        processed_metadata = {}
        for key, value in doc.metadata.items():
            if isinstance(value, list):
                # Convert lists to comma-separated strings
                processed_metadata[key] = ",".join(str(item) for item in value)
            elif isinstance(value, dict):
                # Convert dicts to JSON strings
                import json
                processed_metadata[key] = json.dumps(value)
            else:
                processed_metadata[key] = value
        
        processed_doc = Document(
            page_content=doc.page_content,
            metadata=processed_metadata
        )
        processed_docs.append(processed_doc)
    
    ids = [d.metadata["doc_id"] for d in processed_docs]
    vs.add_documents(documents=processed_docs, ids=ids)
    # Note: langchain-chroma auto-persists to disk, no manual .persist() needed


if __name__ == "__main__":
    import argparse, yaml
    p = argparse.ArgumentParser(description="Ingest cleaned JSONL into Chroma with OpenAI embeddings.")
    p.add_argument("--in", dest="input_path", required=True, help="Path to articles_clean.jsonl")
    p.add_argument("--cfg", dest="cfg_path", default="config/rag.yml", help="Path to rag.yml")
    args = p.parse_args()

    cfg = yaml.safe_load(open(args.cfg_path, "r", encoding="utf-8"))

    ingest_jsonl_to_chroma(
        input_path=args.input_path,
        chroma_dir=cfg["chroma_dir"],
        embedding_model=cfg["embedding_model"],
        chunk_size=cfg["chunk_size"],
        chunk_overlap=cfg["chunk_overlap"],
        min_chars=cfg["min_chars"],
        separators=cfg.get("separators"),
        batch_limit=cfg["batch_limit"],
    )
