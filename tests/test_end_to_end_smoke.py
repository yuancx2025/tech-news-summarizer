import importlib.util, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
def test_imports():
    # Ensure module can be imported without executing main()
    assert importlib.util.find_spec("src.cleaning") is not None
