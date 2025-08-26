# scripts/eval_rag_vs_vanilla.py
from __future__ import annotations
import argparse, json, csv
from pathlib import Path
from typing import Dict, List
from datetime import datetime

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

from model.rag_summarizer import RAGSummarizer, RAGConfig
from model.metrics import rouge_batch, lengths, compression_ratio, Timer

# We try to use your existing summarizers.py; fallback to HF if unavailable
def get_vanilla_summarizer(model_key: str):
    try:
        from model.summarizers import get_summarizer  # your repo function
        summ = get_summarizer(model_key)
        return lambda text: summ(text, max_tokens=180, min_tokens=60)
    except Exception:
        from transformers import pipeline
        hf = "facebook/bart-large-cnn" if model_key.lower() == "bart" else "sshleifer/distilbart-cnn-12-6"
        pipe = pipeline("summarization", model=hf)
        return lambda text: pipe(text, max_length=180, min_length=60, do_sample=False, truncation=True)[0]["summary_text"].strip()


def read_jsonl(path: Path, limit: int | None = None) -> List[dict]:
    items = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip(): continue
            items.append(json.loads(line))
            if limit and len(items) >= limit: break
    return items


def main():
    ap = argparse.ArgumentParser(description="Compare Vanilla vs RAG on a subset and write metrics")
    ap.add_argument("--in", dest="input_jsonl", required=True, help="Processed JSONL")
    ap.add_argument("--db", default="data/processed/news.sqlite", help="SQLite with article_embeddings")
    ap.add_argument("--faiss", default="data/processed/news.faiss", help="FAISS index")
    ap.add_argument("--model", default="bart", help="Vanilla summarizer key (bart/distilbart)")
    ap.add_argument("--embed-model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--k", type=int, default=5, help="Neighbors for RAG")
    ap.add_argument("--n", type=int, default=100, help="Sample size")
    ap.add_argument("--run-name", default="rag_baseline", help="results/<run-name>/")
    ap.add_argument("--id-field", default="id")
    args = ap.parse_args()

    # Validate required input artifacts exist before proceeding
    input_path = Path(args.input_jsonl)
    db_path = Path(args.db)
    faiss_path = Path(args.faiss)
    for p, desc in ((input_path, "input_jsonl"), (db_path, "db"), (faiss_path, "faiss")):
        if not p.exists():
            raise FileNotFoundError(f"{desc} not found: {p}")

    items = read_jsonl(input_path, limit=args.n)
    
    assert items, "No items loaded"

    # Build id->article dict for RAG
    id2article: Dict[int, dict] = {}
    for it in items:
        aid = int(it.get(args.id_field, len(id2article)))
        id2article[aid] = it

    # Init RAG & vanilla
    rag = RAGSummarizer(
        RAGConfig(
            faiss_path=args.faiss,
            sqlite_path=args.db,
            model_name=args.model,
            embed_model=args.embed_model,
            k=args.k,
        ),
        id2article,
    )
    vanilla = get_vanilla_summarizer(args.model)

    out_dir = Path("results") / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs_path = out_dir / "vanilla_vs_rag.jsonl"
    metrics_csv = out_dir / "metrics.csv"
    manifest_path = out_dir / "eval_manifest.json"

    # Run both systems
    preds_v, preds_r, refs_src = [], [], []
    lat_v, lat_r = Timer(), Timer()

    print(f"[eval] Processing {len(items)} articles...")
    
    with pairs_path.open("w", encoding="utf-8") as wf:
        for i, it in enumerate(items):
            if i % 10 == 0:
                print(f"[eval] Progress: {i}/{len(items)} articles processed")
                
            try:
                aid = int(it.get(args.id_field))
                src_text = (it.get("text") or "")
                title = (it.get("title") or "")
                ref = f"{title}\n{src_text}"
                refs_src.append(ref)

                # Vanilla
                lat_v.tic()
                try:
                    v = vanilla(ref[:5000])  # light truncate to keep latency reasonable
                except Exception as e:
                    print(f"[eval] Warning: Vanilla summarization failed for article {aid}: {e}")
                    v = f"[ERROR: {str(e)[:100]}]"
                lat_v.toc()

                # RAG
                lat_r.tic()
                try:
                    r = rag.summarize_article(aid, k=args.k)
                except Exception as e:
                    print(f"[eval] Warning: RAG summarization failed for article {aid}: {e}")
                    r = f"[ERROR: {str(e)[:100]}]"
                lat_r.toc()

                preds_v.append(v)
                preds_r.append(r)

                wf.write(json.dumps({
                    "article_id": aid,
                    "title": title,
                    "source": it.get("source"),
                    "published_at": it.get("published_at"),
                    "vanilla": v,
                    "rag": r
                }, ensure_ascii=False) + "\n")
                
            except Exception as e:
                print(f"[eval] Error processing article {i}: {e}")
                # Add placeholder entries to maintain alignment
                preds_v.append(f"[ERROR: {str(e)[:100]}]")
                preds_r.append(f"[ERROR: {str(e)[:100]}]")
                refs_src.append("")
                continue

    # Metrics (ROUGE against source text as a proxy for recall/coverage)
    rouge_v = rouge_batch(preds_v, refs_src)
    rouge_r = rouge_batch(preds_r, refs_src)
    len_v = lengths(preds_v); len_r = lengths(preds_r)
    comp_v = compression_ratio(refs_src, preds_v)
    comp_r = compression_ratio(refs_src, preds_r)

    # Write CSV summary
    with metrics_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "vanilla", "rag"])
        
        # Handle both ROUGE and fallback metrics
        if "rouge1" in rouge_v and "rouge1" in rouge_r:
            # Standard ROUGE metrics
            for m in ("rouge1", "rouge2", "rougeLsum"):
                if m in rouge_v and m in rouge_r:
                    w.writerow([f"{m}_p", f"{rouge_v[m]['p']:.4f}", f"{rouge_r[m]['p']:.4f}"])
                    w.writerow([f"{m}_r", f"{rouge_v[m]['r']:.4f}", f"{rouge_r[m]['r']:.4f}"])
                    w.writerow([f"{m}_f", f"{rouge_v[m]['f']:.4f}", f"{rouge_r[m]['f']:.4f}"])
        elif "overlap" in rouge_v and "overlap" in rouge_r:
            # Fallback overlap metrics
            w.writerow(["overlap_p", f"{rouge_v['overlap']['p']:.4f}", f"{rouge_r['overlap']['p']:.4f}"])
            w.writerow(["overlap_r", f"{rouge_v['overlap']['r']:.4f}", f"{rouge_r['overlap']['r']:.4f}"])
            w.writerow(["overlap_f", f"{rouge_v['overlap']['f']:.4f}", f"{rouge_r['overlap']['f']:.4f}"])
            if "note" in rouge_v:
                w.writerow(["note", rouge_v["note"], ""])
        
        w.writerow(["len_words_avg", f"{len_v['words_avg']:.1f}", f"{len_r['words_avg']:.1f}"])
        w.writerow(["len_words_p90", f"{len_v['words_p90']:.1f}", f"{len_r['words_p90']:.1f}"])
        w.writerow(["compression_ratio", f"{comp_v:.4f}", f"{comp_r:.4f}"])
        lv = lat_v.stats(); lr = lat_r.stats()
        w.writerow(["latency_ms_avg", f"{lv['ms_avg']:.1f}", f"{lr['ms_avg']:.1f}"])
        w.writerow(["latency_ms_p90", f"{lv['ms_p90']:.1f}", f"{lr['ms_p90']:.1f}"])

    # Manifest
    manifest = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "input_jsonl": str(Path(args.input_jsonl)),
        "n": len(items),
        "k": args.k,
        "vanilla_model": args.model,
        "embed_model": args.embed_model,
        "faiss_path": args.faiss,
        "db_path": args.db,
        "pairs_path": str(pairs_path),
        "metrics_csv": str(metrics_csv),
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"[eval] wrote pairs → {pairs_path}")
    print(f"[eval] wrote metrics → {metrics_csv}")
    print(f"[eval] wrote manifest → {manifest_path}")
    print("[eval] DONE ✅")


if __name__ == "__main__":
    main()
