# src/rag/chains.py
"""LangChain prompt templates and map-reduce summarization chains."""
from __future__ import annotations
from typing import List, Dict, Any, Tuple
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import re

# ---------- LLM factory ----------
def make_llm(model: str = "gpt-4o-mini", temperature: float = 0.0, max_tokens: int = 300):
    return ChatOpenAI(model=model, temperature=temperature, max_tokens=max_tokens)

# ---------- Prompts ----------
MAP_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are a precise, neutral news summarizer. "
               "Write 1–2 factual bullet points. Append a citation like (SOURCE, YYYY-MM-DD) to each bullet."),
    ("human", "Title: {title}\nSource: {source}\nDate: {date}\nURL: {url}\n\n"
              "Chunk:\n{chunk}\n\n"
              "Write 1–2 bullets, each ≤20 words, no fluff.")
])

REDUCE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "Merge bullets into a concise 3–6 bullet digest. Remove duplicates, order by impact then recency. "
               "Keep citations (SOURCE, YYYY-MM-DD). Include one relevant URL once."),
    ("human", "Bullets:\n{bullets}")
])

REASON_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You write one concise, factual reason tailored to the user. No hype, no guarantees."),
    ("human",
     "User profile: {user_profile}\n"
     "Article: {title} | {source} | {date} | tickers={tickers}\n"
     "Lead: {lead}\n"
     "Write ONE sentence (≤30 words) on why this user will care. Mention ≤1 ticker & one angle.")
])

# ---------- Helpers ----------
def _extract_bullets(text: str) -> List[str]:
    lines = [l.strip("•- ").strip() for l in text.strip().splitlines() if l.strip()]
    # Keep lines that look like bullets or short sentences
    bullets = [l for l in lines if len(l) > 0]
    return bullets

def _dedup_bullets(bullets: List[str], jaccard_threshold: float = 0.85) -> List[str]:
    def tokens(s: str) -> set:
        return set(re.findall(r"\w+", s.lower()))
    out: List[str] = []
    seen: List[set] = []
    for b in bullets:
        tb = tokens(b)
        # substring check
        if any(b in x or x in b for x in out):
            continue
        # jaccard dedup
        if any((len(tb & ts) / max(1, len(tb | ts))) >= jaccard_threshold for ts in seen):
            continue
        out.append(b)
        seen.append(tb)
    return out

def _fmt_citation(source: str | None, date: str | None) -> str:
    src = source or "Source"
    dt = (date or "")[:10]
    return f"({src}, {dt})" if dt else f"({src})"

# ---------- Map → Reduce summarization ----------
def map_summarize_per_doc(llm, doc) -> List[str]:
    m = doc.metadata
    res = MAP_PROMPT | llm
    out = res.invoke({
        "title": m.get("title") or "",
        "source": m.get("source_short") or "",
        "date": m.get("published_at") or "",
        "url": m.get("url_canonical") or "",
        "chunk": doc.page_content
    })
    bullets = _extract_bullets(out.content)
    # Guarantee a citation even if model missed it
    cit = _fmt_citation(m.get("source_short"), m.get("published_at"))
    bullets = [b if re.search(r"\)\s*$", b) else f"{b} {cit}" for b in bullets]
    return bullets[:2]

def reduce_merge_bullets(llm, all_bullets: List[str], max_bullets: int = 6) -> List[str]:
    text = "\n".join([f"- {b}" for b in all_bullets])
    out = (REDUCE_PROMPT | llm).invoke({"bullets": text})
    merged = _extract_bullets(out.content)
    merged = _dedup_bullets(merged)
    return merged[:max_bullets]

# ---------- One-line reason ----------
def generate_reason(llm, user_profile: Dict[str, Any], meta: Dict[str, Any]) -> str:
    out = (REASON_PROMPT | llm).invoke({
        "user_profile": user_profile,
        "title": meta.get("title") or "",
        "source": meta.get("source_short") or "",
        "date": meta.get("published_at") or "",
        "tickers": meta.get("tickers") or [],
        "lead": meta.get("lead3") or ""
    })
    # light post-trim
    s = out.content.strip().replace("\n", " ")
    return s[:240]
