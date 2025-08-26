# scripts/rag_batch.py
from __future__ import annotations

import argparse, json, sqlite3
from pathlib import Path
from typing import Dict, List

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

from model.rag_summarizer import RAGSummarizer, RAGConfig


def read_jsonl(path: Path, limit: int | None = None) -> List[dict]:
    items = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            items.append(json.loads(line))
            if limit and len(items) >= limit:
                break
    return items


def main():
    ap = argparse.ArgumentParser(description="Run RAG summarization over a subset of articles")
    ap.add_argument("--in", dest="input_jsonl", required=True, help="Processed JSONL (same one used for embeddings)")
    ap.add_argument("--faiss", default="data/processed/news.faiss", help="FAISS index path")
    ap.add_argument("--db", default="data/processed/news.sqlite", help="SQLite path with article_embeddings")
    ap.add_argument("--model", default="bart", help="Summarizer model key (e.g., bart, distilbart)")
    ap.add_argument("--embed-model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--k", type=int, default=5, help="# neighbors for RAG context")
    ap.add_argument("--n", type=int, default=100, help="How many articles to run")
    ap.add_argument("--run-name", default="rag_baseline", help="Folder in results/ to write outputs")
    ap.add_argument("--id-field", default="id")
    args = ap.parse_args()

    items = read_jsonl(Path(args.input_jsonl), limit=args.n)
    if not items:
        raise SystemExit("No items to process.")
    id_field = args.id_field

    # Build id -> article dict for fast lookup
    id2article: Dict[int, dict] = {}
    for it in items:
        aid = int(it.get(id_field, len(id2article)))
        id2article[aid] = it

    cfg = RAGConfig(
        faiss_path=args.faiss,
        sqlite_path=args.db,
        model_name=args.model,
        embed_model=args.embed_model,
        k=args.k,
    )
    rag = RAGSummarizer(cfg, id2article)

    out_dir = Path("results") / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = out_dir / "summaries.jsonl"

    n_ok = 0
    with out_jsonl.open("w", encoding="utf-8") as wf:
        for it in items:
            aid = int(it.get(id_field))
            try:
                summary = rag.summarize_article(aid, k=args.k)
                if summary:
                    rag.save_summary(aid, summary, k=args.k)
                    record = {
                        "article_id": aid,
                        "title": it.get("title"),
                        "source": it.get("source"),
                        "published_at": it.get("published_at"),
                        "summary_rag": summary,
                        "k": args.k,
                        "model": args.model,
                    }
                    wf.write(json.dumps(record, ensure_ascii=False) + "\n")
                    n_ok += 1
            except Exception as e:
                # keep going; print lightweight error
                print(f"[warn] failed article_id={aid}: {e}")

    print(f"[rag_batch] wrote {n_ok} summaries → {out_jsonl}")
    print(f"[rag_batch] also saved to sqlite → rag_summaries")


if __name__ == "__main__":
    main()
