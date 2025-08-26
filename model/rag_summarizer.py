# model/rag_summarizer.py
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timezone
import numpy as np

# Local imports (repo root on sys.path recommended in your scripts)
from src.faiss_store import read_index, search
from src.embeddings import Embedder, EmbedderConfig
from src.embeddings import combine_title_text

# --- Light abstraction over your existing summarizers -------------------------

class _SummarizerWrapper:
    """
    Tries to use your existing model/summarizers.py; falls back to HF pipeline if unavailable.
    Requires transformers in requirements (you already have it).
    """
    def __init__(self, model_name: str = "bart"):
        try:
            # Expect one of these to exist in your repo
            from model.summarizers import get_summarizer  # type: ignore
            self._summ = get_summarizer(model_name)
            self._kind = "custom"
        except Exception:
            from transformers import pipeline
            hf = "facebook/bart-large-cnn" if model_name.lower() == "bart" else "sshleifer/distilbart-cnn-12-6"
            self._pipe = pipeline("summarization", model=hf)
            self._kind = "hf"

    def summarize(self, text: str, max_tokens: int = 180, min_tokens: int = 60) -> str:
        if self._kind == "custom":
            # Assume your summarizer supports a simple call signature
            return self._summ(text, max_tokens=max_tokens, min_tokens=min_tokens)
        else:
            # HuggingFace pipeline
            out = self._pipe(
                text,
                max_length=max_tokens,
                min_length=min_tokens,
                do_sample=False,
                truncation=True,
            )
            return out[0]["summary_text"].strip()


# --- Utilities ----------------------------------------------------------------

def _simple_sentence_split(s: str, max_len_chars: int = 10000) -> List[str]:
    s = s[:max_len_chars]
    # very light splitter; you already removed NLTK dep
    import re
    chunks = re.split(r"(?<=[.!?])\s+", s)
    return [c.strip() for c in chunks if c.strip()]

def _chunk_text_by_words(s: str, max_words: int = 260) -> List[str]:
    words = s.split()
    out, cur = [], []
    for w in words:
        cur.append(w)
        if len(cur) >= max_words:
            out.append(" ".join(cur))
            cur = []
    if cur:
        out.append(" ".join(cur))
    return out

def _build_context_block(neighbors: List[Tuple[int, float]], id2article: Dict[int, dict], max_per_neighbor_words: int = 80) -> str:
    """
    neighbors: list of (article_id, score)
    """
    lines = []
    for aid, score in neighbors:
        art = id2article.get(aid)
        if not art:
            continue
        title = (art.get("title") or "").strip()
        src   = (art.get("source") or "").strip()
        date  = (art.get("published_at") or "")[:10]
        body  = (art.get("text") or "")
        snippet = " ".join(_chunk_text_by_words(body, max_words=max_per_neighbor_words)[:1])
        header = f"- [{date}] {src} — {title}"
        lines.append(header)
        if snippet:
            lines.append(f"  {snippet}")
    return "\n".join(lines).strip()

def _map_reduce_summary(summarizer: _SummarizerWrapper, article_text: str, max_words_per_chunk: int = 280) -> str:
    # Map
    chunks = _chunk_text_by_words(article_text, max_words=max_words_per_chunk)
    if not chunks:
        return ""
    partials = [summarizer.summarize(c, max_tokens=150, min_tokens=50) for c in chunks[:6]]  # cap chunks for speed
    merged = "\n".join("- " + p for p in partials if p)

    # Reduce
    final_prompt = f"""You are a news editor. Summarize the article in 4–6 crisp bullet points.

Content (already partially summarized in bullets):
{merged}
"""
    return summarizer.summarize(final_prompt, max_tokens=180, min_tokens=60)

# --- RAG Summarizer -----------------------------------------------------------

@dataclass
class RAGConfig:
    faiss_path: str = "data/processed/news.faiss"
    sqlite_path: str = "data/processed/news.sqlite"
    model_name: str = "bart"
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    k: int = 5

class RAGSummarizer:
    """
    Retrieval-augmented summarization:
      - retrieve K neighbors via FAISS
      - build a short context block
      - run map-reduce summary on (context + article)
      - save to SQLite
    """
    def __init__(self, cfg: RAGConfig, id2article: Dict[int, dict]):
        self.cfg = cfg
        self.index = read_index(cfg.faiss_path)
        self.embedder = Embedder(EmbedderConfig(model_name=cfg.embed_model, normalize=True, batch_size=64))
        self.summarizer = _SummarizerWrapper(cfg.model_name)
        self.id2article = id2article  # article_id -> dict (title, text, source, published_at, ...)
        # build idx<->id maps from sqlite
        self._idx2id, self._id2idx = self._load_idx_maps(cfg.sqlite_path)

    @staticmethod
    def _load_idx_maps(sqlite_path: str) -> Tuple[List[int], Dict[int, int]]:
        conn = sqlite3.connect(sqlite_path)
        try:
            cur = conn.execute("SELECT article_id, idx FROM article_embeddings ORDER BY idx ASC")
            rows = cur.fetchall()
        finally:
            conn.close()
        idx2id = [None] * len(rows)
        id2idx: Dict[int, int] = {}
        for aid, idx in rows:
            idx2id[idx] = int(aid)
            id2idx[int(aid)] = int(idx)
        return idx2id, id2idx

    def _neighbors_for_article(self, article_id: int, topk: int) -> List[Tuple[int, float]]:
        """
        Uses the stored embedding for this article (via idx) and searches FAISS.
        """
        if article_id not in self._id2idx:
            # fallback: embed on the fly
            art = self.id2article.get(article_id, {})
            qtext = combine_title_text(art.get("title"), art.get("text"), max_chars=1200)
            qvec = self.embedder.encode([qtext])[0]
        else:
            # we don't need the raw vector if FAISS is already built from the same matrix;
            # but FAISS requires a query vector. Use the embedder to re-embed the article text to be safe.
            art = self.id2article.get(article_id, {})
            qtext = combine_title_text(art.get("title"), art.get("text"), max_chars=1200)
            qvec = self.embedder.encode([qtext])[0]

        scores, idxs = search(self.index, qvec.astype(np.float32), topk=topk + 1)
        # Map FAISS indices -> article ids; drop self if present
        out: List[Tuple[int, float]] = []
        for sc, idx in zip(scores, idxs):
            if idx < 0:
                continue
            aid = self._idx2id[idx]
            if aid is None:
                continue
            if int(aid) == int(article_id):
                continue
            out.append((int(aid), float(sc)))
            if len(out) >= topk:
                break
        return out

    def summarize_article(self, article_id: int, k: Optional[int] = None) -> str:
        k = k or self.cfg.k
        art = self.id2article.get(article_id, {})
        title = (art.get("title") or "").strip()
        body  = (art.get("text") or "")
        if not body and not title:
            return ""

        neighbors = self._neighbors_for_article(article_id, topk=k)
        ctx = _build_context_block(neighbors, self.id2article, max_per_neighbor_words=80)

        prompt = f"""You are a professional tech & finance news editor.
Use the context to improve coverage and precision, but avoid repetition.

Context (related articles):
{ctx or '(no additional context)'}
---
Article:
Title: {title}
Body:
{body[:4000]}

Write a brief with 4–6 bullet points. Be factual, objective, and concise.
"""
        # Map-reduce on the article itself (body), then fold context implicitly through the prompt
        brief = _map_reduce_summary(self.summarizer, prompt, max_words_per_chunk=260)
        return brief.strip()

    # --- persistence ----------------------------------------------------------

    def save_summary(self, article_id: int, summary: str, k: int):
        conn = sqlite3.connect(self.cfg.sqlite_path)
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS rag_summaries (
                    article_id INTEGER,
                    k          INTEGER,
                    model      TEXT,
                    summary    TEXT,
                    created_at TEXT
                )
                """
            )
            conn.execute(
                """
                INSERT INTO rag_summaries(article_id, k, model, summary, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    int(article_id),
                    int(k),
                    self.cfg.model_name,
                    summary,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            conn.commit()
        finally:
            conn.close()
