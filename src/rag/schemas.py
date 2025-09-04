# src/rag/schemas.py
from __future__ import annotations
from typing import List, Optional, Literal
from pydantic import BaseModel

# ----- Summarization -----
class SummarizeRequest(BaseModel):
    mode: Literal["article", "topic"] = "topic"
    query: Optional[str] = None          # topic mode
    article_url: Optional[str] = None    # article mode (canonical URL)
    article_id: Optional[str] = None     # alternative to URL if known
    k: int = 8
    days_back: int = 60
    tickers: Optional[List[str]] = None
    sources: Optional[List[str]] = None

class SummaryBullet(BaseModel):
    text: str
    citation: str
    url: Optional[str] = None

class SummarizeResponse(BaseModel):
    bullets: List[SummaryBullet]

# ----- Recommendation -----
class UserProfile(BaseModel):
    tickers: Optional[List[str]] = None
    sections: Optional[List[str]] = None
    sources: Optional[List[str]] = None
    days_back: int = 60

class RecommendRequest(BaseModel):
    user: UserProfile
    interest_text: Optional[str] = None  # free-text preference
    k: int = 5

class RecommendItem(BaseModel):
    title: str
    url: str
    source: str
    published_at: str
    reason: str

class RecommendResponse(BaseModel):
    items: List[RecommendItem]
