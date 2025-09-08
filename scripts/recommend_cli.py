# scripts/recommend_cli.py
import argparse, yaml, os, sys
from typing import List
from src.rag.tool import RAGTool
from src.rag.schemas import RecommendRequest, UserProfile
from dotenv import load_dotenv

def _csv_list(s: str) -> List[str] | None:
    return [x.strip() for x in s.split(",") if x.strip()] if s else None

def main():
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY environment variable is required. Please set it in .env file or environment.")
    
    ap = argparse.ArgumentParser("Recommend articles with one-line reasons")
    ap.add_argument("--cfg", default="config/rag.yml")
    ap.add_argument("--tickers", default="", help="Comma-separated tickers (optional)")
    ap.add_argument("--sources", default="", help="Comma-separated sources (optional)")
    ap.add_argument("--sections", default="", help="Comma-separated sections (optional)")
    ap.add_argument("--days-back", type=int, default=None, help="Override profile recency window")
    ap.add_argument("--interest", default=None, help="Free-text interest (optional)")
    ap.add_argument("--k", type=int, default=None, help="Top-N (override)")
    args = ap.parse_args()

    if not os.path.exists(args.cfg):
        print(f"[ERR] config not found: {args.cfg}", file=sys.stderr); sys.exit(2)
    cfg = yaml.safe_load(open(args.cfg, "r", encoding="utf-8"))

    k = args.k if args.k is not None else cfg.get("recommend", {}).get("k", 5)
    days = args.days_back if args.days_back is not None else cfg.get("recommend", {}).get("days_back_default", 60)

    rag = RAGTool(cfg)

    prof = UserProfile(
        tickers=[t.upper() for t in _csv_list(args.tickers) or []] or None,
        sources=_csv_list(args.sources),
        sections=_csv_list(args.sections),
        days_back=days,
    )
    req = RecommendRequest(user=prof, interest_text=args.interest, k=k)
    res = rag.recommend(req)

    if not res.items:
        print("[INFO] No recommendations."); return

    print("RECOMMENDATIONS\n---------------")
    for i, it in enumerate(res.items, 1):
        print(f"{i:02d}. {it.title} ({it.source}, {it.published_at})")
        print(f"    {it.reason}")
        if it.url: print(f"    {it.url}")

if __name__ == "__main__":
    main()
