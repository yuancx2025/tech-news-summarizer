# model/rag_summarizer.py
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timezone
import numpy as np
import os

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
        self._kind = "none"
        self._summ = None
        self._pipe = None
        
        # Check if we're on macOS and force fallback mode to avoid MPS issues
        import platform
        if platform.system() == "Darwin":
            print(f"[_SummarizerWrapper] macOS detected, using fallback mode to avoid MPS issues")
            self._kind = "fallback"
            return
        
        # Try custom summarizer first
        try:
            from model.summarizers import get_summarizer  # type: ignore
            self._summ = get_summarizer(model_name)
            self._kind = "custom"
            print(f"[_SummarizerWrapper] Using custom summarizer: {model_name}")
        except Exception as e:
            print(f"[_SummarizerWrapper] Custom summarizer failed: {e}")
            
        # Fallback to HuggingFace pipeline with robust error handling
        if self._kind == "none":
            try:
                # Disable MPS for macOS compatibility
                os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
                
                from transformers import pipeline
                
                # Use smaller, more compatible models
                if model_name.lower() == "bart":
                    hf_model = "sshleifer/distilbart-cnn-12-6"  # Smaller, more stable
                elif model_name.lower() == "distilbart":
                    hf_model = "sshleifer/distilbart-cnn-12-6"
                else:
                    hf_model = "sshleifer/distilbart-cnn-12-6"
                
                print(f"[_SummarizerWrapper] Loading HF model: {hf_model}")
                
                # Force CPU to avoid MPS issues
                self._pipe = pipeline(
                    "summarization", 
                    model=hf_model,
                    device=-1  # Force CPU
                )
                self._kind = "hf"
                print(f"[_SummarizerWrapper] HF pipeline loaded successfully")
                
            except Exception as e:
                print(f"[_SummarizerWrapper] HF pipeline failed: {e}")
                # Final fallback: simple text truncation
                self._kind = "fallback"
                print(f"[_SummarizerWrapper] Using fallback summarizer (text truncation)")

    def summarize(self, text: str, max_tokens: int = 180, min_tokens: int = 60) -> str:
        if self._kind == "custom" and self._summ:
            try:
                return self._summ(text, max_tokens=max_tokens, min_tokens=min_tokens)
            except Exception as e:
                print(f"[_SummarizerWrapper] Custom summarizer failed: {e}")
                # Fall through to other methods
                
        if self._kind == "hf" and self._pipe:
            try:
                # Truncate text to avoid memory issues
                max_input_length = 1024
                if len(text) > max_input_length:
                    text = text[:max_input_length]
                
                out = self._pipe(
                    text,
                    max_length=max_tokens,
                    min_length=min_tokens,
                    do_sample=False,
                    truncation=True,
                )
                return out[0]["summary_text"].strip()
            except Exception as e:
                print(f"[_SummarizerWrapper] HF pipeline failed: {e}")
                # Fall through to fallback
                
        # Fallback: simple text truncation
        if self._kind == "fallback" or self._kind == "none":
            return self._fallback_summarize(text, max_tokens)
            
        # If all else fails, return truncated text
        return text[:max_tokens * 4] + "..." if len(text) > max_tokens * 4 else text
    
    def _fallback_summarize(self, text: str, max_tokens: int) -> str:
        """Simple fallback summarizer that provides basic text summarization"""
        # Remove any prompt text that might be in the input
        if "You are a" in text and "news editor" in text:
            # Extract just the article content
            if "Article:" in text:
                text = text.split("Article:")[-1]
            elif "Body:" in text:
                text = text.split("Body:")[-1]
        
        # Clean up the text
        text = text.strip()
        if not text:
            return "No content available for summarization."
        
        # Split into sentences and take the most important ones
        sentences = text.split('. ')
        if len(sentences) <= 3:
            return text
        
        # Simple heuristic: take first sentence (title/headline) + a few key sentences
        result = [sentences[0]]  # Start with first sentence
        
        # Add middle sentences (often contain key information)
        mid_point = len(sentences) // 2
        if mid_point < len(sentences):
            result.append(sentences[mid_point])
        
        # Add last sentence if it's not too long
        if len(sentences) > 1 and len(sentences[-1]) < 100:
            result.append(sentences[-1])
        
        # Limit total length
        summary = '. '.join(result)
        if len(summary) > max_tokens * 4:
            summary = summary[:max_tokens * 4]
            if not summary.endswith('.'):
                summary += '...'
        
        return summary

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
        try:
            conn = sqlite3.connect(sqlite_path)
            try:
                cur = conn.execute("SELECT article_id, idx FROM article_embeddings ORDER BY idx ASC")
                rows = cur.fetchall()
            finally:
                conn.close()
            
            if not rows:
                print(f"[RAGSummarizer] Warning: No rows found in {sqlite_path}")
                return [], {}
                
            idx2id = [None] * len(rows)
            id2idx: Dict[int, int] = {}
            for aid, idx in rows:
                idx2id[idx] = int(aid)
                id2idx[int(aid)] = int(idx)
            return idx2id, id2idx
            
        except Exception as e:
            print(f"[RAGSummarizer] Warning: Could not load index maps from {sqlite_path}: {e}")
            print(f"[RAGSummarizer] Will use fallback mode (re-embedding for all queries)")
            return [], {}

    def _neighbors_for_article(self, article_id: int, topk: int) -> List[Tuple[int, float]]:
        """
        Uses the stored embedding for this article (via idx) and searches FAISS.
        Falls back to re-embedding if index maps are not available.
        """
        # If we have index maps, try to use stored embeddings
        if self._id2idx and article_id in self._id2idx:
            try:
                # Get the stored vector from FAISS index
                idx = self._id2idx[article_id]
                if hasattr(self.index, 'reconstruct'):
                    # Try to reconstruct the vector from the index
                    qvec = self.index.reconstruct(idx).astype(np.float32)
                else:
                    # Fallback: re-embed the article text
                    art = self.id2article.get(article_id, {})
                    qtext = combine_title_text(art.get("title"), art.get("text"), max_chars=1200)
                    qvec = self.embedder.encode([qtext])[0]
            except Exception as e:
                print(f"[RAGSummarizer] Warning: Could not reconstruct vector for article {article_id}: {e}")
                # Fall through to re-embedding
                art = self.id2article.get(article_id, {})
                qtext = combine_title_text(art.get("title"), art.get("text"), max_chars=1200)
                qvec = self.embedder.encode([qtext])[0]
        else:
            # Fallback: embed on the fly
            art = self.id2article.get(article_id, {})
            qtext = combine_title_text(art.get("title"), art.get("text"), max_chars=1200)
            qvec = self.embedder.encode([qtext])[0]

        try:
            scores, idxs = search(self.index, qvec.astype(np.float32), topk=topk + 1)
        except Exception as e:
            print(f"[RAGSummarizer] Warning: FAISS search failed: {e}")
            return []

        # Map FAISS indices -> article ids; drop self if present
        out: List[Tuple[int, float]] = []
        for sc, idx in zip(scores, idxs):
            if idx < 0:
                continue
                
            # If we have index maps, use them; otherwise use the index directly
            if self._idx2id and idx < len(self._idx2id):
                aid = self._idx2id[idx]
                if aid is None:
                    continue
                if int(aid) == int(article_id):
                    continue
            else:
                # Fallback: use the index directly as article ID
                aid = int(idx)
                if aid == int(article_id):
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

        # If using fallback summarizer, provide a simpler approach
        if self.summarizer._kind == "fallback":
            # For fallback mode, just summarize the article text directly
            article_text = f"{title}\n\n{body[:3000]}"  # Limit body length
            return self.summarizer.summarize(article_text, max_tokens=150, min_tokens=50)

        # Full RAG approach for other summarizer types
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
        try:
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
        except Exception as e:
            print(f"[RAGSummarizer] Warning: Could not save summary to database: {e}")
            # Continue without saving to database
