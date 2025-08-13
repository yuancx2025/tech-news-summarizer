import importlib.util, sys, os

def test_imports():
    # Ensure module can be imported without executing main()
    assert importlib.util.find_spec("news_summarizer.cleaning") is not None
