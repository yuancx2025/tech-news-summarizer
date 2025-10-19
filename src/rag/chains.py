# src/rag/chains.py
"""LangChain prompt templates and map-reduce summarization chains."""
from __future__ import annotations
from typing import List, Dict, Any, Tuple, Literal, Optional
import os
import json
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
import re

# ---------- LLM factory ----------
def make_llm(
    model: str = "gemini-2.5-flash",
    temperature: float = 0.0,
    max_tokens: int = 300,
    provider: Literal["openai", "gemini"] = "gemini",
    api_key: Optional[str] = None,
):
    """Create LLM instance. Defaults to Gemini 2.5 Flash for summarization."""
    if provider == "gemini":
        key = api_key or os.getenv("GOOGLE_API_KEY")
        if not key:
            raise RuntimeError(
                "GOOGLE_API_KEY is required for Gemini LLM usage. Set the environment "
                "variable or pass api_key explicitly."
            )
        return ChatGoogleGenerativeAI(
            model=model,
            temperature=temperature,
            max_output_tokens=max_tokens,
            google_api_key=key,
        )
    else:
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

QA_MAP_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You extract timeline events that directly answer the user's question."
        " Respond ONLY with JSON like {\"events\": [...]}. Each event should include"
        " date (YYYY-MM-DD), summary (≤40 words, factual), source, url, and citation"
        " text such as (SOURCE, YYYY-MM-DD). Use provided metadata when helpful."
        " Return {\"events\": []} if nothing is relevant.",
    ),
    (
        "human",
        "Question: {question}\n"
        "Title: {title}\nSource: {source}\nDate: {date}\nURL: {url}\n\n"
        "Content:\n{chunk}\n",
    ),
])

QA_REDUCE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You synthesize trend-aware answers from a dated event timeline."
        " Write 2-4 sentences that address the user's question, weaving in"
        " the key developments. Reference citations exactly as provided in the"
        " timeline (e.g., append (SOURCE, YYYY-MM-DD) at the end of facts)."
        " If there is insufficient evidence, say so.",
    ),
    (
        "human",
        "Question: {question}\n"
        "Timeline:\n{timeline}",
    ),
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


def _safe_json_loads(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        text = text.strip()
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                pass
    return {"events": []}

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


# ---------- QA map-reduce ----------
def map_events_from_doc(llm, question: str, doc) -> List[Dict[str, Any]]:
    meta = doc.metadata
    raw = (QA_MAP_PROMPT | llm).invoke({
        "question": question,
        "title": meta.get("title") or "",
        "source": meta.get("source_short") or "",
        "date": meta.get("published_at") or "",
        "url": meta.get("url_canonical") or "",
        "chunk": doc.page_content,
    })
    data = _safe_json_loads(raw.content)
    events = data.get("events") or []
    normalized: List[Dict[str, Any]] = []
    for ev in events:
        summary = (ev or {}).get("summary") or ""
        if not summary.strip():
            continue
        date = (ev or {}).get("date") or meta.get("published_at") or ""
        source = (ev or {}).get("source") or meta.get("source_short") or ""
        url = (ev or {}).get("url") or meta.get("url_canonical") or ""
        citation = (ev or {}).get("citation")
        if not citation:
            citation = _fmt_citation(source, date)
        normalized.append(
            {
                "date": date[:10] if isinstance(date, str) else str(date),
                "summary": summary.strip(),
                "source": source or None,
                "url": url or None,
                "citation": citation,
            }
        )
    return normalized


def reduce_answer_from_events(llm, question: str, events: List[Dict[str, Any]]) -> Tuple[str, List[str]]:
    if not events:
        return "I could not find enough evidence to answer that question confidently.", []

    lines = []
    for idx, ev in enumerate(events, start=1):
        citation = ev.get("citation") or ""
        date = ev.get("date") or "Unknown"
        summary = ev.get("summary") or ""
        lines.append(f"{idx}. {date} – {summary} {citation}".strip())

    joined = "\n".join(lines)
    out = (QA_REDUCE_PROMPT | llm).invoke({"question": question, "timeline": joined})
    answer = out.content.strip()
    found = re.findall(r"\([^\)]+\d{4}-\d{2}-\d{2}\)", answer)
    citations = list(dict.fromkeys(found))
    if not citations:
        citations = [ev.get("citation") for ev in events if ev.get("citation")]
        citations = list(dict.fromkeys(citations))
    return answer, citations


def run_qa_chain(
    llm_map,
    llm_reduce,
    question: str,
    docs: List[Any],
    *,
    max_events: int = 6,
) -> Tuple[str, List[Dict[str, Any]], List[str]]:
    events: List[Dict[str, Any]] = []
    for doc in docs:
        events.extend(map_events_from_doc(llm_map, question, doc))

    # Deduplicate by (date, summary)
    deduped: List[Dict[str, Any]] = []
    seen = set()
    for ev in events:
        key = (ev.get("date"), ev.get("summary"))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(ev)

    deduped.sort(key=lambda e: (e.get("date") or "9999-99-99"))
    trimmed = deduped[:max_events]
    answer, citations = reduce_answer_from_events(llm_reduce, question, trimmed)
    return answer, trimmed, citations
