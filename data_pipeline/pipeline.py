# data_pipeline/pipeline.py
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional, List

from dotenv import load_dotenv

# Import pipeline functions directly (no shell-out)
from data_pipeline.scrape import load_config, crawl_domain, write_jsonl
from data_pipeline.clean import load_jsonl as load_clean_jsonl, normalize_record, dedupe_exact, write_jsonl as write_clean_jsonl, write_csv
from src.rag.ingest import ingest_jsonl_to_chroma

def run_scrape(config_path: str, out_jsonl: str, per_request_sleep: float = 0.4) -> None:
    cfg = load_config(config_path)
    sources = cfg.get("sources", [])
    if not sources:
        raise ValueError(f"No sources found in {config_path}")

    all_rows = []
    for src in sources:
        domain = src["domain"]
        limit = int(src.get("limit", 30))
        lang = src.get("language", "en")
        print(f"[SCRAPE] {domain} (limit={limit}, lang={lang})")
        rows = crawl_domain(domain, limit=limit, lang=lang, per_request_sleep=per_request_sleep)
        print(f"[SCRAPE]   +{len(rows)} rows")
        all_rows.extend(rows)

    if not all_rows:
        print("[SCRAPE] No rows collected.")
        return

    Path(out_jsonl).parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_jsonl, all_rows)
    print(f"[SCRAPE] Wrote JSONL → {out_jsonl}")

def run_clean(in_jsonl: str, out_jsonl: str, out_csv: Optional[str] = None, 
              drop_non_en: bool = False) -> None:
    """
    Clean raw articles: normalize, dedupe, and export to JSONL/CSV.
    Simplified signature using defaults from config.
    """
    Path(out_jsonl).parent.mkdir(parents=True, exist_ok=True)
    if out_csv:
        Path(out_csv).parent.mkdir(parents=True, exist_ok=True)

    print(f"[CLEAN] Processing {in_jsonl}")
    
    # Streaming pipeline
    def normalize_stream():
        for rec in load_clean_jsonl(in_jsonl):
            norm = normalize_record(rec)
            if not norm:
                continue
            if drop_non_en and norm.get("language") != "en":
                continue
            yield norm
    
    # Dedupe and write JSONL
    cleaned = dedupe_exact(normalize_stream())
    n = write_clean_jsonl(out_jsonl, cleaned)
    print(f"[CLEAN] Wrote {n} rows → {out_jsonl}")
    
    # Optional CSV export
    if out_csv:
        cleaned_for_csv = dedupe_exact(normalize_stream())
        fields = ["id", "url", "source", "title", "published_at", "language"]
        n_csv = write_csv(out_csv, cleaned_for_csv, fields)
        print(f"[CLEAN] Wrote {n_csv} rows → {out_csv}")

def run_index(in_clean_jsonl: str, rag_cfg_path: str) -> None:
    """Index cleaned JSONL into Chroma using config file."""
    import yaml
    
    cfg_path = Path(rag_cfg_path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {rag_cfg_path}")
    
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    
    # Validate required keys
    for key in ["chroma_dir", "embedding_model", "chunk_size", "chunk_overlap", "min_chars", "batch_limit"]:
        if key not in cfg:
            raise KeyError(f"[INDEX] Missing '{key}' in {rag_cfg_path}")
    
    chroma_dir = Path(cfg["chroma_dir"])
    chroma_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[INDEX] Config: {rag_cfg_path}")
    print(f"[INDEX] Chroma dir: {chroma_dir}")

    ingest_jsonl_to_chroma(
        input_path=in_clean_jsonl,
        chroma_dir=str(chroma_dir),
        embedding_model=cfg["embedding_model"],
        chunk_size=int(cfg["chunk_size"]),
        chunk_overlap=int(cfg["chunk_overlap"]),
        min_chars=int(cfg["min_chars"]),
        separators=cfg.get("separators"),
        batch_limit=int(cfg["batch_limit"]),
    )
    print("[INDEX] Indexed into Chroma.")

def main():
    load_dotenv()

    ap = argparse.ArgumentParser(description="End-to-end pipeline: scrape → clean → index")
    sub = ap.add_subparsers(dest="cmd", required=True)

    # scrape
    sp = sub.add_parser("scrape")
    sp.add_argument("--config", default="config/feeds.yml")
    sp.add_argument("--out-jsonl", default="data/raw/articles.jsonl")
    sp.add_argument("--sleep", type=float, default=0.4)

    # clean
    cp = sub.add_parser("clean")
    cp.add_argument("--in-jsonl", required=True, help="Raw JSONL input")
    cp.add_argument("--out-jsonl", required=True, help="Cleaned JSONL output")
    cp.add_argument("--out-csv", default=None, help="Optional CSV export")
    cp.add_argument("--drop-non-en", action="store_true", help="Drop non-English articles")

    # index
    ip = sub.add_parser("index")
    ip.add_argument("--input", required=True, help="Clean JSONL (e.g., data/processed/articles_clean.jsonl)")
    ip.add_argument("--cfg", default="config/rag.yml", help="RAG config file")

    # all
    ap_all = sub.add_parser("all")
    ap_all.add_argument("--feeds", default="config/feeds.yml", help="Feed sources config")
    ap_all.add_argument("--raw-jsonl", default="data/raw/articles.jsonl")
    ap_all.add_argument("--clean-jsonl", default="data/processed/articles_clean.jsonl")
    ap_all.add_argument("--clean-csv", default="data/processed/articles_clean.csv")
    ap_all.add_argument("--rag-cfg", default="config/rag.yml")

    args = ap.parse_args()

    if args.cmd == "scrape":
        run_scrape(args.config, args.out_jsonl, per_request_sleep=args.sleep)

    elif args.cmd == "clean":
        run_clean(
            in_jsonl=args.in_jsonl,
            out_jsonl=args.out_jsonl,
            out_csv=args.out_csv,
            drop_non_en=args.drop_non_en,
        )

    elif args.cmd == "index":
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY is required for indexing.")
        run_index(args.input, args.cfg)

    elif args.cmd == "all":
        # 1) scrape
        run_scrape(args.feeds, args.raw_jsonl)
        # 2) clean
        run_clean(
            in_jsonl=args.raw_jsonl,
            out_jsonl=args.clean_jsonl,
            out_csv=args.clean_csv,
        )
        # 3) index
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY is required for indexing.")
        run_index(args.clean_jsonl, args.rag_cfg)

if __name__ == "__main__":
    main()
