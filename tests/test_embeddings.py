import json
import sqlite3
import numpy as np
from types import SimpleNamespace
from pathlib import Path

import src.embeddings as embmod


def test_combine_and_norm():
    s = embmod.combine_title_text("Hello", "world " * 100, max_chars=50, title_weight=2)
    assert "Hello" in s and len(s) <= 60  # title + truncated body

    X = np.array([[3.0, 4.0], [0.0, 0.0]], dtype=np.float32)
    Y = embmod.ensure_unit_norm(X)
    assert np.allclose(np.linalg.norm(Y[0]), 1.0, atol=1e-6)
    assert np.allclose(Y[1], np.zeros(2))


def test_build_embeddings_smoke(tmp_path, monkeypatch):
    # Tiny JSONL
    data = [
        {"id": 1, "title": "A", "text": "alpha beta gamma"},
        {"id": 2, "title": "B", "text": "beta gamma delta"},
        {"id": 3, "title": "C", "text": "gamma delta epsilon"},
    ]
    in_jsonl = tmp_path / "preprocessed.jsonl"
    with in_jsonl.open("w", encoding="utf-8") as f:
        for row in data:
            f.write(json.dumps(row) + "\n")

    # Monkeypatch Embedder.encode to avoid downloading models
    def fake_encode(self, texts):
        # Simple bag-of-words over 3 tokens → 3D vectors, then L2-normalize
        vocab = {"alpha":0, "beta":1, "gamma":2, "delta":1, "epsilon":2}
        mat = []
        for t in texts:
            vec = np.zeros(3, dtype=np.float32)
            for tok in t.split():
                if tok in vocab:
                    vec[vocab[tok]] += 1.0
            mat.append(vec)
        M = np.vstack(mat)
        return embmod.ensure_unit_norm(M)

    monkeypatch.setattr(embmod.Embedder, "encode", fake_encode)

    # Import and run builder
    from scripts.build_embeddings import build
    out_npy = tmp_path / "embeddings.npy"
    out_db = tmp_path / "news.sqlite"
    args = SimpleNamespace(
        input=str(in_jsonl),
        out_npy=str(out_npy),
        out_db=str(out_db),
        model_name="dummy",
        batch_size=2,
        max_chars=500,
        title_weight=1,
        id_field="id",
        title_field="title",
        text_field="text",
        limit=None,
        normalize=True,
        force_post_normalize=False,
    )
    build(args)

    # Check outputs
    assert out_npy.exists() and out_db.exists()
    X = np.load(out_npy)
    assert X.shape == (3, 3)
    # DB rows
    conn = sqlite3.connect(out_db)
    try:
        n, = conn.execute("SELECT COUNT(*) FROM article_embeddings").fetchone()
        assert n == 3
        dim, = conn.execute("SELECT d FROM article_embeddings LIMIT 1").fetchone()
        assert dim == 3
    finally:
        conn.close()
