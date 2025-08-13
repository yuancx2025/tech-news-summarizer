# src/schema.py
SCHEMA = """
CREATE TABLE IF NOT EXISTS articles (
  id INTEGER PRIMARY KEY,
  url TEXT UNIQUE,
  title TEXT,
  authors TEXT,
  published TEXT,
  text TEXT,
  summary TEXT,
  keywords TEXT,
  content_hash TEXT,
  sentiment TEXT,
  fetched_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_published ON articles(published);
CREATE INDEX IF NOT EXISTS idx_content_hash ON articles(content_hash);
"""