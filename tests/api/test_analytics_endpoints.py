import json
import importlib
import sys
import types
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def analytics_app(tmp_path, monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    monkeypatch.setenv("ANALYTICS_DIR", str(tmp_path))

    dummy_module = types.SimpleNamespace(
        GoogleGenerativeAIEmbeddings=lambda model, google_api_key: object(),
        ChatGoogleGenerativeAI=lambda **_: object(),
    )
    monkeypatch.setitem(sys.modules, "langchain_google_genai", dummy_module)

    overview = {
        "generated_at": "2024-05-05T00:00:00Z",
        "sector_sentiment": [
            {"sector": "Technology", "date": "2024-05-04", "sentiment": 0.3, "article_count": 4},
            {"sector": "Technology", "date": "2024-05-06", "sentiment": -0.1, "article_count": 2},
        ],
        "ticker_mentions": [
            {"ticker": "AAPL", "mentions": 5},
            {"ticker": "MSFT", "mentions": 3},
        ],
    }
    cooccurrence = [
        {"source": "AAPL", "target": "MSFT", "weight": 2},
        {"source": "AAPL", "target": "NVDA", "weight": 1},
    ]
    Path(tmp_path, "overview.json").write_text(json.dumps(overview), encoding="utf-8")
    Path(tmp_path, "ticker_cooccurrence.json").write_text(json.dumps(cooccurrence), encoding="utf-8")

    module = importlib.import_module("src.api.api_main")
    return TestClient(module.app)


def test_analytics_overview_filters_date_and_ticker(analytics_app):
    resp = analytics_app.get("/analytics/overview", params={"start": "2024-05-05", "tickers": "AAPL"})
    assert resp.status_code == 200
    data = resp.json()
    assert all(row["date"] >= "2024-05-05" for row in data["sector_sentiment"])
    assert data["ticker_mentions"] == [{"ticker": "AAPL", "mentions": 5}]


def test_analytics_cooccurrence_limit_and_filter(analytics_app):
    resp = analytics_app.get("/analytics/cooccurrence", params={"tickers": "NVDA", "limit": 1})
    assert resp.status_code == 200
    data = resp.json()
    assert data["pairs"] == [{"source": "AAPL", "target": "NVDA", "weight": 1}]
