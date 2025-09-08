# scripts/summarize_cli.py
import argparse, yaml, os, sys
from typing import List
from src.rag.tool import RAGTool
from src.rag.schemas import SummarizeRequest

def _csv_list(s: str) -> List[str] | None:
    return [x.strip() for x in s.split(",") if x.strip()] if s else None

def main():
    ap = argparse.ArgumentParser("Summarize (topic or article) via RAG")
    ap.add_argument("--cfg", default="config/rag.yml")
    ap.add_argument("--mode", choices=["topic","article"], default="topic")
    ap.add_argument("--query", help="Topic query (topic mode)")
    ap.add_argument("--article-url", help="Article canonical URL (article mode)")
    ap.add_argument("--article-id", help="Article ID (article mode)")
    ap.add_argument("--tickers", default="", help="Comma-separated tickers (optional)")
    ap.add_argument("--sources", default="", help="Comma-separated sources (optional)")
    ap.add_argument("--k", type=int, default=None, help="Final k (override)")
    args = ap.parse_args()

    if not os.path.exists(args.cfg):
        print(f"[ERR] config not found: {args.cfg}", file=sys.stderr); sys.exit(2)
    cfg = yaml.safe_load(open(args.cfg, "r", encoding="utf-8"))

    # optional k override
    if args.k is not None:
        cfg.setdefault("summarize", {})["k_final"] = args.k

    rag = RAGTool(cfg)

    req = SummarizeRequest(
        mode=args.mode,
        query=args.query,
        article_url=args.article_url,
        article_id=args.article_id,
        k=cfg.get("summarize", {}).get("k_final", 8),
        tickers=[t.upper() for t in _csv_list(args.tickers) or []] or None,
        sources=_csv_list(args.sources),
    )

    res = rag.summarize(req)
    if not res.bullets:
        print("[INFO] No bullets returned."); return
    print("SUMMARY\n-------")
    for b in res.bullets:
        line = f"- {b.text}"
        if b.url: line += f"  {b.url}"
        print(line)

if __name__ == "__main__":
    main()
