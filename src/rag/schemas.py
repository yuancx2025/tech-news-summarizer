# src/rag/schemas.py
from __future__ import annotations
from datetime import date, datetime
from typing import List, Optional, Literal, Union

from pydantic import BaseModel, Field

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


# ----- Question Answering -----
class TemporalFilter(BaseModel):
    """Optional explicit or natural-language bounds for timeline queries."""

    natural_language: Optional[str] = Field(
        default=None,
        description="Free-text hint such as 'since July' or 'last quarter'.",
    )
    start_date: Optional[Union[date, datetime]] = Field(
        default=None, description="Explicit inclusive start date for events."
    )
    end_date: Optional[Union[date, datetime]] = Field(
        default=None, description="Explicit inclusive end date for events."
    )
    days_back: Optional[int] = Field(
        default=None,
        description="Fallback recency window when no explicit dates are provided.",
    )


class QuestionRequest(BaseModel):
    question: str
    days_back: int = Field(60, ge=1, description="Default window when no dates are provided.")
    temporal_filter: Optional[TemporalFilter] = None
    tickers: Optional[List[str]] = None
    sources: Optional[List[str]] = None
    language: str = "en"
    max_events: int = Field(
        6,
        ge=1,
        le=20,
        description="Maximum number of supporting timeline events to return.",
    )
    context_k: int = Field(
        16,
        ge=4,
        le=60,
        description="Number of context chunks to retrieve before filtering events.",
    )


class TimelineItem(BaseModel):
    date: Optional[str] = None
    summary: str
    source: Optional[str] = None
    url: Optional[str] = None
    citation: Optional[str] = None


class QuestionResponse(BaseModel):
    answer: str
    timeline: List[TimelineItem]
    citations: List[str]
