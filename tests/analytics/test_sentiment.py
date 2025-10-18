import pytest

from src.analytics.sentiment import bucket_sentiment


@pytest.mark.parametrize(
    "score,expected",
    [(-0.2, "negative"), (0.0, "neutral"), (0.2, "positive")],
)
def test_bucket_sentiment(score, expected):
    assert bucket_sentiment(score) == expected
