from datetime import datetime

from src.analytics.catalog import detect_tickers_from_text, normalize_metadata


def test_normalize_metadata_parses_strings():
    meta = {
        "tickers": "AAPL, msft ,",
        "sectors": ["Technology", "Cloud"],
        "published_at": "2024-05-02T15:30:00Z",
    }
    normalized = normalize_metadata(meta)
    assert normalized["tickers"] == ["AAPL", "MSFT"]
    assert normalized["sectors"] == ["Cloud", "Technology"]
    assert isinstance(normalized["published_at"], datetime)


def test_detect_tickers_respects_lookup():
    text = "Apple (AAPL) outperforms while MSFT consolidates." \
        " Meanwhile, Random words should be ignored."
    detected = detect_tickers_from_text(text, allowed=["AAPL", "META"])
    assert detected == ["AAPL"]
