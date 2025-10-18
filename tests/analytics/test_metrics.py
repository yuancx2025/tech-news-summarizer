import pandas as pd

from src.analytics.metrics import compute_cooccurrence


def test_compute_cooccurrence_counts_pairs():
    df = pd.DataFrame(
        [
            {"article_id": "1", "tickers": ["AAPL", "MSFT", "NVDA"]},
            {"article_id": "2", "tickers": ["AAPL", "MSFT"]},
            {"article_id": "3", "tickers": ["AAPL"]},
        ]
    )
    co = compute_cooccurrence(df)
    assert not co.empty
    # Expect combinations: (AAPL, MSFT):2, (AAPL, NVDA):1, (MSFT, NVDA):1
    pair_counts = {tuple(row[:2]): row[2] for row in co.to_numpy()}
    assert pair_counts[("AAPL", "MSFT")] == 2
    assert pair_counts[("AAPL", "NVDA")] == 1
    assert pair_counts[("MSFT", "NVDA")] == 1
