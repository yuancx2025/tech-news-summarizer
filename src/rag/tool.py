# src/rag/tool.py
"""High-level RAGTool class orchestrating retrieval, summarization, and recommendations."""
from __future__ import annotations

import os
from typing import List, Dict, Any, Optional
import yaml
from collections import defaultdict

from langchain_chroma import Chroma
from langchain.schema import Document

from src.embeddings import get_embeddings
from src.rag.schemas import (
    SummarizeRequest,
    SummarizeResponse,
    SummaryBullet,
    RecommendRequest,
    RecommendResponse,
    RecommendItem,
    QuestionRequest,
    QuestionResponse,
    TimelineItem,
)
from src.rag.retriever import retrieve_with_config
from src.rag.ranking import (
    score_and_rank_candidates, filter_near_duplicate_titles
)
from src.rag.chains import (
    make_llm,
    map_summarize_per_doc,
    reduce_merge_bullets,
    generate_reason,
    run_qa_chain,
)
from src.rag.ingest import _canonicalize_url
from src.rag.time_utils import resolve_temporal_filter

# --------- RAGTool ----------
class RAGTool:
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.gemini_api_key = cfg.get("gemini_api_key") or os.getenv("GOOGLE_API_KEY")
        if not self.gemini_api_key:
            raise RuntimeError(
                "GOOGLE_API_KEY is required for Gemini provider. Set the environment variable "
                "or add 'gemini_api_key' to config/rag.yml."
            )
        self.emb = get_embeddings(
            provider="gemini",
            model_name=cfg.get("embedding_model", "models/text-embedding-004"),
            api_key=self.gemini_api_key,
        )
        self.vs = Chroma(persist_directory=cfg["chroma_dir"], embedding_function=self.emb)
        self.llm_summarize = make_llm(
            model=cfg.get("llm_model", "gemini-2.5-flash"),
            temperature=0.0,
            max_tokens=300,
            provider="gemini",
            api_key=self.gemini_api_key,
        )
        self.llm_reason = make_llm(
            model=cfg.get("llm_model", "gemini-2.5-flash"),
            temperature=0.0,
            max_tokens=250,
            provider="gemini",
            api_key=self.gemini_api_key,
        )
        self.llm_qa_map = make_llm(
            model=cfg.get("llm_model", "gemini-2.5-flash"),
            temperature=0.0,
            max_tokens=400,
            provider="gemini",
            api_key=self.gemini_api_key,
        )
        self.llm_qa_reduce = make_llm(
            model=cfg.get("llm_model", "gemini-2.5-flash"),
            temperature=0.1,
            max_tokens=600,
            provider="gemini",
            api_key=self.gemini_api_key,
        )

    # ---- Summarization ----
    def summarize(self, req: SummarizeRequest) -> SummarizeResponse:
        if req.mode == "article":
            docs = self._fetch_article_chunks(article_id=req.article_id, url=req.article_url)
            # Select up to 2 representative chunks: first + a later one (diversity by position)
            selected = self._select_article_chunks(docs, max_per_article=2, k_final=req.k)
        else:
            # topic mode via retriever (MMR + filters + auto-expand)
            docs = retrieve_with_config(
                req.query or "",
                self.cfg,
                tickers=req.tickers,
                sources=req.sources
            )
            # group by article and keep top 1–2 chunks/article
            selected = self._group_and_select(docs, per_article=2, k_final=req.k)

        # If nothing was selected/fetched, return empty result (avoid reduce step)
        if not selected:
            return SummarizeResponse(bullets=[])

        # Map → reduce
        map_bullets: List[str] = []
        for d in selected:
            map_bullets.extend(map_summarize_per_doc(self.llm_summarize, d))

        # If map produced nothing, return empty result
        if not map_bullets:
            return SummarizeResponse(bullets=[])

        merged = reduce_merge_bullets(
            self.llm_summarize, map_bullets, max_bullets=self.cfg.get("summarize", {}).get("max_bullets", 6)
        )

        # Choose one canonical URL (most recent among selected)
        canon_url = None
        if selected:
            selected_sorted = sorted(selected, key=lambda d: d.metadata.get("published_at") or "", reverse=True)
            canon_url = selected_sorted[0].metadata.get("url_canonical")

        bullets = [SummaryBullet(text=b, citation="", url=canon_url) for b in merged]
        return SummarizeResponse(bullets=bullets)

    def _fetch_article_chunks(self, *, article_id: Optional[str], url: Optional[str]) -> List[Document]:
        where: Dict[str, Any] = {}
        if article_id:
            where["article_id"] = article_id
        elif url:
            # Canonicalize to match how URLs are stored during ingestion
            canon = _canonicalize_url(url)
            where["url_canonical"] = canon
        else:
            return []

        res = self.vs._collection.get(where=where, include=["documents", "metadatas"])  # type: ignore[attr-defined]
        docs: List[Document] = []
        for doc_id, content, meta in zip(res.get("ids", []), res.get("documents", []), res.get("metadatas", [])):
            # Ensure doc_id is present in metadata for ranking compatibility
            meta = dict(meta or {})
            meta["doc_id"] = doc_id
            docs.append(Document(page_content=content or "", metadata=meta))
        # sort by chunk_idx
        docs.sort(key=lambda d: d.metadata.get("chunk_idx", 0))
        return docs

    def _select_article_chunks(self, docs: List[Document], max_per_article: int = 2, k_final: int = 8) -> List[Document]:
        # For a single article: pick first and a later chunk far enough away
        if not docs:
            return []
        picks = [docs[0]]
        if len(docs) > 1:
            # pick a chunk ~ mid or last for diversity
            picks.append(docs[len(docs)//2] if len(docs) >= 3 else docs[-1])
        return picks[:max_per_article]

    def _group_and_select(self, docs: List[Document], per_article: int = 2, k_final: int = 8) -> List[Document]:
        groups: Dict[str, List[Document]] = defaultdict(list)
        for d in docs:
            aid = d.metadata.get("article_id") or d.metadata.get("doc_id", "").split("::")[0]
            groups[aid].append(d)
        selected: List[Document] = []
        for aid, arr in groups.items():
            arr = sorted(arr, key=lambda d: d.metadata.get("chunk_idx", 0))
            selected.extend(arr[:per_article])
            if len(selected) >= k_final:
                break
        return selected[:k_final]

    # ---- Recommendations ----
    def recommend(self, req: RecommendRequest) -> RecommendResponse:
        user = req.user
        query = req.interest_text or self._default_interest_query(user)

        # Retrieve a diverse set (MMR); then de-dup press-release near-dups by title
        docs = retrieve_with_config(query, self.cfg, tickers=user.tickers, sources=user.sources)
        docs = filter_near_duplicate_titles(docs, threshold=0.85)

        # Score and rank by sim + recency
        rcfg = self.cfg.get("recommend", {})
        ranked = score_and_rank_candidates(
            query=query,
            docs=docs,
            vs=self.vs,
            emb=self.emb,
            sim_weight=rcfg.get("sim_weight", 0.65),
            recency_weight=rcfg.get("recency_weight", 0.35),
            recency_half_life_days=rcfg.get("recency_half_life_days", 7),
        )

        # Take top-k and produce one-line reasons
        topk = ranked[: req.k]
        items: List[RecommendItem] = []
        for d in topk:
            m = d.metadata
            reason = generate_reason(self.llm_reason, user.dict(), m)
            items.append(RecommendItem(
                title=m.get("title") or "Untitled",
                url=m.get("url_canonical") or "",
                source=m.get("source_short") or "",
                published_at=m.get("published_at") or "",
                reason=reason
            ))
        return RecommendResponse(items=items)

    def _default_interest_query(self, user) -> str:
        bits: List[str] = []
        if user.tickers: bits.append(" ".join([f"${t}" for t in user.tickers]))
        if user.sections: bits.append(" ".join(user.sections))
        if user.sources: bits.append(" ".join(user.sources))
        phrase = " ".join(bits) if bits else "technology and finance"
        return f"Notable developments and analysis for {phrase}"

    # ---- Question Answering ----
    def answer_question(self, req: QuestionRequest) -> QuestionResponse:
        start_date, end_date, days_back = resolve_temporal_filter(
            req.temporal_filter,
            default_days_back=req.days_back,
        )

        docs = retrieve_with_config(
            req.question,
            self.cfg,
            tickers=req.tickers,
            sources=req.sources,
            start_date=start_date,
            end_date=end_date,
            days_back_start=days_back,
            k_final=req.context_k,
            fetch_k=max(req.context_k, req.context_k * 2),
        )

        if not docs:
            return QuestionResponse(answer="No supporting articles were found.", timeline=[], citations=[])

        answer_text, events, citations = run_qa_chain(
            self.llm_qa_map,
            self.llm_qa_reduce,
            req.question,
            docs,
            max_events=req.max_events,
        )

        # Normalize timeline ordering & shape
        def _event_key(ev: Dict[str, Any]) -> str:
            date = ev.get("date") or "9999-99-99"
            return date

        events_sorted = sorted(events, key=_event_key)

        timeline_items = [
            TimelineItem(
                date=ev.get("date"),
                summary=ev.get("summary", ""),
                source=ev.get("source"),
                url=ev.get("url"),
                citation=ev.get("citation"),
            )
            for ev in events_sorted
        ]

        all_citations = list(dict.fromkeys([c for c in citations if c]))
        if not all_citations:
            all_citations = [
                ev.citation for ev in timeline_items if ev.citation
            ]
            all_citations = list(dict.fromkeys(all_citations))

        return QuestionResponse(
            answer=answer_text,
            timeline=timeline_items,
            citations=all_citations,
        )
