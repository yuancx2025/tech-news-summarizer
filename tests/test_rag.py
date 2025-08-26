import json
import sqlite3
from pathlib import Path
import numpy as np
import types

# We will monkeypatch:
# - Embedder.encode -> deterministic tiny vectors
# - read_index/search -> search over our small matrix
# - _SummarizerWrapper.summarize -> echo-like deterministic output

def test_rag_end_to_end(tmp_path, monkeypatch):
    # ---------- tiny corpus ----------
    items = [
        {"id": 1, "title": "OpenAI launches model", "text": "alpha beta gamma"},
        {"id": 2, "title": "Google releases tool", "text": "beta gamma delta"},
        {"id": 3, "title": "Finance report", "text": "earnings revenue guidance"},
    ]
    in_jsonl = tmp_path / "pre.jsonl"
    with in_jsonl.open("w", encoding="utf-8") as f:
        for r in items:
            f.write(json.dumps(r) + "\n")

    # Simple 4D embeddings for 3 docs (unit normalized)
    vocab = {"alpha":0, "beta":1, "gamma":2, "delta":3, "earnings":0, "revenue":1, "guidance":2}
    def text_to_vec(s: str) -> np.ndarray:
        v = np.zeros(4, dtype=np.float32)
        for tok in s.split():
            if tok in vocab:
                v[vocab[tok]] += 1.0
        n = np.linalg.norm(v) or 1.0
        return (v / n).astype(np.float32)

    embs = np.vstack([text_to_vec(it["text"]) for it in items])  # (3,4)

    # ---------- sqlite mapping ----------
    db = tmp_path / "news.sqlite"
    conn = sqlite3.connect(db)
    conn.execute("""
        CREATE TABLE article_embeddings(
            article_id INTEGER PRIMARY KEY,
            idx INTEGER, d INTEGER, npy_path TEXT,
            model_name TEXT, normalized INTEGER, created_at TEXT
        )
    """)
    for i, it in enumerate(items):
        conn.execute(
            "INSERT INTO article_embeddings(article_id, idx, d, npy_path, model_name, normalized, created_at) VALUES (?,?,?,?,?,?,?)",
            (it["id"], i, 4, "X.npy", "dummy", 1, "now"),
        )
    conn.commit(); conn.close()

    # ---------- monkeypatch embeddings + faiss ----------
    import src.embeddings as E
    monkeypatch.setattr(E.Embedder, "encode", lambda self, texts: np.vstack([text_to_vec(t) for t in texts]))

    # fake faiss "index": just hold the matrix; implement read_index & search
    import src.faiss_store as F

    class DummyIndex:
        def __init__(self, X): self.X = X

    def fake_read_index(path): return DummyIndex(embs)

    def fake_search(index, qvec, topk=5):
        if qvec.ndim == 1:
            q = qvec / (np.linalg.norm(qvec) or 1.0)
        else:
            q = qvec[0] / (np.linalg.norm(qvec[0]) or 1.0)
        sims = (index.X @ q.astype(np.float32))
        order = np.argsort(-sims)
        return sims[order][:topk], order[:topk]

    monkeypatch.setattr(F, "read_index", fake_read_index)
    monkeypatch.setattr(F, "search", fake_search)

    # ---------- monkeypatch summarizer to be deterministic ----------
    import model.rag_summarizer as R

    class DummySumm:
        def __init__(self, *args, **kwargs): 
            self._kind = "fallback"  # Add the _kind attribute
        def summarize(self, text, max_tokens=180, min_tokens=60):
            # Return first 100 chars to simulate a "summary"
            return ("SUMMARY: " + text[:100]).strip()

    monkeypatch.setattr(R, "_SummarizerWrapper", DummySumm)

    # ---------- run RAG ----------
    id2article = {it["id"]: it for it in items}
    cfg = R.RAGConfig(faiss_path="ignored.faiss", sqlite_path=str(db), model_name="dummy", embed_model="dummy", k=2)
    rag = R.RAGSummarizer(cfg, id2article)

    s = rag.summarize_article(1, k=2)
    assert s.startswith("SUMMARY:")
    assert len(s) > 10

    # save to sqlite
    rag.save_summary(1, s, k=2)
    conn = sqlite3.connect(db)
    try:
        conn.execute("SELECT 1 FROM rag_summaries LIMIT 1").fetchone()
    finally:
        conn.close()
