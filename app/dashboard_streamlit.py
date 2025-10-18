"""Streamlit dashboard for analytics visualizations."""
from __future__ import annotations

import os
from datetime import date, timedelta
from typing import Dict, Optional

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

DEFAULT_API = os.getenv("API_BASE_URL", "http://localhost:8000")


@st.cache_data(show_spinner=False)
def _build_heatmap_matrix(co_df: pd.DataFrame) -> pd.DataFrame:
    tickers = sorted(set(co_df["source"]).union(co_df["target"]))
    base = pd.DataFrame(0, index=tickers, columns=tickers, dtype=float)
    for row in co_df.itertuples(index=False):
        a, b, weight = row.source, row.target, getattr(row, "weight", 0)
        base.loc[a, b] += weight
        base.loc[b, a] += weight
    return base


def _build_params(start: Optional[date], end: Optional[date], tickers: list[str]) -> Dict[str, str]:
    params: Dict[str, str] = {}
    if start:
        params["start"] = start.isoformat()
    if end:
        params["end"] = end.isoformat()
    if tickers:
        params["tickers"] = ",".join(sorted({t.upper() for t in tickers if t}))
    return params


st.set_page_config(page_title="Analytics Dashboard", layout="wide")
st.title("📊 News Analytics Dashboard")

with st.sidebar:
    st.header("Settings")
    api_base = st.text_input("API Base URL", value=DEFAULT_API)
    default_end = date.today()
    default_start = default_end - timedelta(days=30)
    start_date = st.date_input("Start date", value=default_start)
    end_date = st.date_input("End date", value=default_end)

@st.cache_data(show_spinner=False)
def fetch_overview(api: str, params: Dict[str, str]) -> Dict:
    resp = requests.get(f"{api}/analytics/overview", params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


@st.cache_data(show_spinner=False)
def fetch_cooccurrence(api: str, params: Dict[str, str]) -> Dict:
    resp = requests.get(f"{api}/analytics/cooccurrence", params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()

# Load base overview for ticker options
try:
    base_overview = fetch_overview(api_base, {})
except Exception as exc:
    st.error(f"Failed to load analytics overview: {exc}")
    base_overview = {"ticker_mentions": []}

all_tickers = sorted({row["ticker"] for row in base_overview.get("ticker_mentions", []) if row.get("ticker")})
selected_tickers = st.sidebar.multiselect("Filter tickers", options=all_tickers)

params = _build_params(start_date, end_date, selected_tickers)
try:
    overview = fetch_overview(api_base, params)
except Exception as exc:
    st.error(f"Failed to refresh overview data: {exc}")
    overview = {"sector_sentiment": [], "ticker_mentions": []}

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Sector Sentiment (Daily Average)")
    sector_df = pd.DataFrame(overview.get("sector_sentiment", []))
    if sector_df.empty:
        st.info("No sector sentiment data available for the selected filters.")
    else:
        sector_df["date"] = pd.to_datetime(sector_df["date"])
        fig = px.area(
            sector_df,
            x="date",
            y="sentiment",
            color="sector",
            line_group="sector",
            title="Average sentiment by sector",
        )
        fig.update_layout(margin=dict(l=40, r=20, t=60, b=40))
        st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Top Mentioned Tickers")
    mention_df = pd.DataFrame(overview.get("ticker_mentions", []))
    if mention_df.empty:
        st.info("No ticker mentions found for the selected filters.")
    else:
        mention_df = mention_df.sort_values("mentions", ascending=False).head(20)
        fig = px.bar(mention_df, x="mentions", y="ticker", orientation="h", title="Mentions")
        fig.update_layout(yaxis=dict(categoryorder="total ascending"), margin=dict(l=100, r=20, t=60, b=40))
        st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.subheader("Ticker Co-occurrence Heatmap")
co_params = {"limit": "1000"}
if selected_tickers:
    co_params["tickers"] = ",".join(selected_tickers)
try:
    cooccurrence = fetch_cooccurrence(api_base, co_params)
except Exception as exc:
    st.error(f"Failed to fetch co-occurrence data: {exc}")
    cooccurrence = {"pairs": []}
co_df = pd.DataFrame(cooccurrence.get("pairs", []))
if co_df.empty:
    st.info("No co-occurrence pairs available. Try expanding the date range or removing filters.")
else:
    matrix_df = _build_heatmap_matrix(co_df)
    fig = px.imshow(matrix_df, text_auto=True, aspect="auto", color_continuous_scale="Blues")
    fig.update_layout(margin=dict(l=40, r=20, t=40, b=40))
    st.plotly_chart(fig, use_container_width=True)
