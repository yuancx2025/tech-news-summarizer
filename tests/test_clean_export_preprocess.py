import os
import sys
import sqlite3

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from scripts.clean_export_preprocess import fetch_from_sqlite, chunkify_by_tokens


def test_fetch_from_sqlite_filters_and_dedup(tmp_path):
    db_path = tmp_path / "sample.sqlite"
    con = sqlite3.connect(db_path)
    con.execute(
        """
        CREATE TABLE articles (
            id TEXT, url TEXT, title TEXT, published TEXT,
            text TEXT, source TEXT, content_hash TEXT
        )
        """
    )
    long_text = "A" * 400
    con.executemany(
        "INSERT INTO articles VALUES (?,?,?,?,?,?,?)",
        [
            ("1", "http://example.com/a", "Title A", "2024-01-01", long_text, "example.com", "hash1"),
            ("2", "http://example.com/a", "Title A updated", "2024-01-02", long_text, "example.com", "hash1"),
            ("3", "http://example.com/short", "Short", "2024-01-03", "short", "example.com", "hash3"),
            ("4", "http://other.com/b", "Title B", "2024-01-02", long_text, "other.com", "hash4"),
        ],
    )
    con.commit()
    con.close()

    df = fetch_from_sqlite(str(db_path), "articles", None, None, ["example.com"], 300)
    assert len(df) == 1
    row = df.iloc[0]
    assert row["id"] == "2"
    assert row["title"] == "Title A updated"
    assert row["source"] == "example.com"
    assert row["published"].endswith("Z")


def test_chunkify_by_tokens_word_fallback():
    text = "one two three four five six seven eight nine ten"
    chunks = list(chunkify_by_tokens(text, max_len=5, overlap=2))
    assert chunks == [
        "one two three four five",
        "four five six seven eight",
        "seven eight nine ten",
        "ten",
    ]