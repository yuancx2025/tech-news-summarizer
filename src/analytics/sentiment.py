"""Sentiment scoring utilities for analytics."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

try:  # pragma: no cover - fallback for test environments without vader
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
except ImportError:  # pragma: no cover
    class SentimentIntensityAnalyzer:  # type: ignore[override]
        """Lightweight fallback if vaderSentiment is unavailable."""

        def polarity_scores(self, text: str) -> dict:
            lower = (text or "").lower()
            positive = sum(lower.count(token) for token in ["gain", "beat", "outperform", "growth"])
            negative = sum(lower.count(token) for token in ["loss", "miss", "decline", "slump"])
            compound = 0.0
            if positive or negative:
                compound = (positive - negative) / max(positive + negative, 1)
            return {"compound": compound}


@dataclass
class SentimentAnalyzer:
    """Wrapper around VADER for finance news sentiment."""

    analyzer: SentimentIntensityAnalyzer = SentimentIntensityAnalyzer()

    def score_articles(
        self,
        articles: pd.DataFrame,
        *,
        id_column: str = "article_id",
        text_column: str = "text",
    ) -> pd.DataFrame:
        """Return sentiment scores for each article."""

        rows = []
        for row in articles.itertuples(index=False):
            article_id = getattr(row, id_column)
            text = getattr(row, text_column) or ""
            scores = self.analyzer.polarity_scores(text)
            compound = scores.get("compound", 0.0)
            rows.append(
                {
                    id_column: article_id,
                    "sentiment": compound,
                    "sentiment_label": bucket_sentiment(compound),
                }
            )
        return pd.DataFrame(rows)

    def load_or_score(
        self,
        articles: pd.DataFrame,
        cache_path: str | Path,
        *,
        id_column: str = "article_id",
        text_column: str = "text",
    ) -> pd.DataFrame:
        """Load cached sentiment scores if available, otherwise compute and persist."""

        cache = Path(cache_path)
        if cache.exists():
            return pd.read_parquet(cache)
        cache.parent.mkdir(parents=True, exist_ok=True)
        scored = self.score_articles(articles, id_column=id_column, text_column=text_column)
        scored.to_parquet(cache, index=False)
        return scored


def bucket_sentiment(score: float) -> str:
    """Discretize compound VADER scores into qualitative buckets."""

    if score >= 0.05:
        return "positive"
    if score <= -0.05:
        return "negative"
    return "neutral"
