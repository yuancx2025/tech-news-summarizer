"""Metric aggregation for analytics endpoints."""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Dict, List

import pandas as pd


@dataclass
class AnalyticsMetricsBuilder:
    articles: pd.DataFrame
    sentiment: pd.DataFrame
    output_dir: Path

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if "article_id" not in self.articles.columns:
            raise KeyError("articles DataFrame missing 'article_id'")
        if "article_id" not in self.sentiment.columns:
            raise KeyError("sentiment DataFrame missing 'article_id'")

        if "published_at" in self.articles.columns:
            self.articles["published_at"] = pd.to_datetime(self.articles["published_at"], utc=True, errors="coerce")

    def build(self) -> Dict[str, pd.DataFrame]:
        merged = self.articles.merge(self.sentiment, on="article_id", how="left")
        sector = compute_sector_sentiment(merged)
        mentions = compute_ticker_mentions(merged)
        cooccurrence = compute_cooccurrence(merged)
        return {
            "sector_sentiment": sector,
            "ticker_mentions": mentions,
            "ticker_cooccurrence": cooccurrence,
        }

    def persist(self) -> Dict[str, Path]:
        frames = self.build()
        timestamp = datetime.utcnow().isoformat() + "Z"
        paths: Dict[str, Path] = {}

        overview_payload = {
            "generated_at": timestamp,
            "sector_sentiment": frames["sector_sentiment"].to_dict(orient="records"),
            "ticker_mentions": frames["ticker_mentions"].to_dict(orient="records"),
        }
        overview_path = self.output_dir / "overview.json"
        overview_path.write_text(json.dumps(overview_payload, default=str), encoding="utf-8")
        paths["overview_json"] = overview_path

        for key, df in frames.items():
            parquet_path = self.output_dir / f"{key}.parquet"
            json_path = self.output_dir / f"{key}.json"
            df.to_parquet(parquet_path, index=False)
            json_path.write_text(df.to_json(orient="records", date_format="iso"), encoding="utf-8")
            paths[f"{key}_parquet"] = parquet_path
            paths[f"{key}_json"] = json_path

        paths["generated_at"] = timestamp  # type: ignore[assignment]
        return paths


def compute_sector_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["sector", "date", "sentiment", "article_count"])

    exploded = df.explode("sectors")
    exploded = exploded[exploded["sectors"].notnull()]
    if exploded.empty:
        return pd.DataFrame(columns=["sector", "date", "sentiment", "article_count"])

    exploded["date"] = exploded["published_at"].dt.date
    grouped = (
        exploded.groupby(["sectors", "date"], dropna=True)
        .agg(sentiment=("sentiment", "mean"), article_count=("article_id", "nunique"))
        .reset_index()
    )
    grouped = grouped.rename(columns={"sectors": "sector"}).sort_values(["sector", "date"])
    return grouped


def compute_ticker_mentions(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["ticker", "mentions"])
    exploded = df.explode("tickers")
    exploded = exploded[exploded["tickers"].notnull()]
    if exploded.empty:
        return pd.DataFrame(columns=["ticker", "mentions"])
    grouped = (
        exploded.groupby("tickers", dropna=True)["article_id"].nunique().reset_index(name="mentions")
    )
    grouped = grouped.rename(columns={"tickers": "ticker"}).sort_values("mentions", ascending=False)
    return grouped


def compute_cooccurrence(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["source", "target", "weight"])

    pairs: List[Dict[str, str | int]] = []
    for row in df.itertuples(index=False):
        tickers = sorted({t for t in getattr(row, "tickers", []) if t})
        if len(tickers) < 2:
            continue
        for a, b in combinations(tickers, 2):
            pairs.append({"source": a, "target": b, "weight": 1})

    if not pairs:
        return pd.DataFrame(columns=["source", "target", "weight"])

    pair_df = pd.DataFrame(pairs)
    grouped = (
        pair_df.groupby(["source", "target"], as_index=False)["weight"].sum().sort_values("weight", ascending=False)
    )
    return grouped
