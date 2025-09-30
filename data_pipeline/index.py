# data_pipeline/index.py
"""
Index cleaned articles into Chroma (or your configured vector store).

Reads a cleaned JSONL (from clean.py), performs chunking during ingestion, and
adds useful metadata for RAG (article_id, chunk_idx, n_chunks, published_at, source, url, title).
"""
from __future__ import annotations

import argparse, os, json
from pathlib import Path
from dotenv import load_dotenv

# We rely on project ingestion util that handles chunking + embedding.
# It should accept the parameters we pass below.
try:
    from src.rag.ingest import ingest_jsonl_to_chroma  # type: ignore
except Exception as e:
    raise RuntimeError("Could not import src.rag.ingest.ingest_jsonl_to_chroma. "
                       "Ensure your PYTHONPATH includes the project root.") from e

def sanity_check(input_path: str, sample_size: int = 5):
    """Verify cleaned schema has key fields by sampling multiple records."""
    required = {"id", "title", "text", "url", "published_at"}
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
                if n >= sample_size:
                    break
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON at line {i}: {e}")
    if n == 0:
        raise ValueError("Input appears empty or unreadable.")
    print(f"[index] Validated {n} sample records")

def main():
    load_dotenv()
    ap = argparse.ArgumentParser("Index cleaned news into Chroma")
    ap.add_argument("--input", dest="input_path", required=True, help="Cleaned JSONL (e.g., data/processed/articles_clean.jsonl)")
    ap.add_argument("--config", default="config/rag.yml", help="RAG config with indexing params")
    args = ap.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY environment variable is required. Set it in .env or shell.")

    sanity_check(args.input_path, sample_size=5)
    
    # Load config
    import yaml
    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {args.config}")
    
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    
    # Extract params from config
    chroma_dir = cfg.get("chroma_dir", "data/vdb/chroma")
    embedding_model = cfg.get("embedding_model", "text-embedding-3-small")
    chunk_size = int(cfg.get("chunk_size", 900))
    chunk_overlap = int(cfg.get("chunk_overlap", 120))
    min_chars = int(cfg.get("min_chars", 380))
    batch_limit = int(cfg.get("batch_limit", 1500))
    separators = cfg.get("separators")
    
    Path(chroma_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"[index] Config: {args.config}")
    print(f"[index] Embedding model: {embedding_model}")
    print(f"[index] Chunk size: {chunk_size}, overlap: {chunk_overlap}")

    # The ingestion utility should handle chunking and inject chunk metadata.
    ingest_jsonl_to_chroma(
        input_path=args.input_path,
        chroma_dir=chroma_dir,
        embedding_model=embedding_model,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        min_chars=min_chars,
        separators=separators,
        batch_limit=batch_limit,
    )
    print("[index] Ingestion complete.")

if __name__ == "__main__":
    main()

