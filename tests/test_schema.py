import sqlite3, os, tempfile
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.schema import SCHEMA

def test_schema_creates_tables():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "news.sqlite")
        con = sqlite3.connect(path)
        with con:
            con.executescript(SCHEMA)
        cur = con.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='articles';")
        assert cur.fetchone()[0] == "articles"
        # columns exist
        cols = [r[1] for r in con.execute("PRAGMA table_info(articles);")]
        for c in ["url", "title", "text", "content_hash", "sentiment"]:
            assert c in cols
        con.close()
