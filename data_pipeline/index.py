# data_pipeline/index.py
"""
Index cleaned articles into Chroma (or your configured vector store).

Reads a cleaned JSONL (from clean.py), performs chunking during ingestion, and
adds useful metadata for RAG (article_id, chunk_idx, n_chunks, published_at, source, url, title).
"""
from __future__ import annotations

import argparse, os, json
from dotenv import load_dotenv

# We rely on project ingestion util that handles chunking + embedding.
# It should accept the parameters we pass below.
try:
    from src.rag.ingest import ingest_jsonl_to_chroma  # type: ignore
except Exception as e:
    raise RuntimeError("Could not import src.rag.ingest.ingest_jsonl_to_chroma. "
                       "Ensure your PYTHONPATH includes the project root.") from e

def sanity_check(input_path: str):
    # Verify cleaned schema has key fields
    required = {"id","title","text","url","published_at"}
    n = 0
    with open(input_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
                n += 1
                missing = required - set(obj.keys())
                if missing:
                    raise ValueError(f"Record {i} missing keys: {missing}")
                break  # only sample 1 record for speed
            except json.JSONDecodeError:
                continue
    if n == 0:
        raise ValueError("Input appears empty or unreadable.")

def main():
    load_dotenv()
    ap = argparse.ArgumentParser("Index cleaned news into Chroma")
    ap.add_argument("--input", dest="input_path", required=True, help="data/processed/articles_clean.jsonl")
    ap.add_argument("--chroma-dir", default="data/chroma")
    ap.add_argument("--embedding-model", default="text-embedding-3-large")
    ap.add_argument("--chunk-size", type=int, default=400)
    ap.add_argument("--chunk-overlap", type=int, default=60)
    ap.add_argument("--min-chars", type=int, default=200)
    ap.add_argument("--separators", nargs="*", default=None)
    ap.add_argument("--batch-limit", type=int, default=96)
    args = ap.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY environment variable is required. Set it in .env or shell.")

    sanity_check(args.input_path)

    # The ingestion utility should handle chunking and inject chunk metadata.
    ingest_jsonl_to_chroma(
        input_path=args.input_path,
        chroma_dir=args.chroma_dir,
        embedding_model=args.embedding_model,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        min_chars=args.min_chars,
        separators=args.separators,
        batch_limit=args.batch_limit,
    )
    print("[index] Ingestion complete.")

if __name__ == "__main__":
    main()

