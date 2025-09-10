# data_pipeline/pipeline.py
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import Optional, List, Dict
import os

from dotenv import load_dotenv

# import scraper helpers to avoid an extra process
from data_pipeline.scrape import load_config, crawl_domain, write_jsonl, MIN_CHARS, validate_row  # validate_row imported from src.cleaning via scrape
# index function is already a proper library function
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

def run_clean(in_jsonl: str, out_csv: str, out_jsonl: str, out_pre_jsonl: str,
              tokenizer_model: str = "facebook/bart-large-cnn",
              truncate_to: int = 1000,
              min_chars: int = 300,
              min_words: int = 80,
              sentencer: str = "nltk",
              make_chunks: bool = False,
              chunk_tokens: int = 800,
              chunk_overlap: int = 120,
              chunks_out: str = "data/chunks/chunks_{date}.jsonl",
              date_from: Optional[str] = None,
              date_to: Optional[str] = None,
              sources: Optional[List[str]] = None) -> None:
    """
    Delegate to data_pipeline.clean CLI to avoid duplicating logic.
    """
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    Path(out_jsonl).parent.mkdir(parents=True, exist_ok=True)
    Path(out_pre_jsonl).parent.mkdir(parents=True, exist_ok=True)
    if make_chunks:
        Path(chunks_out.replace("{date}", "1970-01-01")).parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "python", "-m", "data_pipeline.clean",
        "--in-jsonl", in_jsonl,
        "--out-csv", out_csv,
        "--out-jsonl", out_jsonl,
        "--out-pre-jsonl", out_pre_jsonl,
        "--tokenizer-model", tokenizer_model,
        "--truncate-to", str(truncate_to),
        "--min-chars", str(min_chars),
        "--min-words", str(min_words),
        "--sentencer", sentencer,
    ]
    if date_from:
        cmd += ["--date-from", date_from]
    if date_to:
        cmd += ["--date-to", date_to]
    if sources:
        cmd += ["--sources"] + sources
    if make_chunks:
        cmd += ["--make-chunks", "--chunk-tokens", str(chunk_tokens), "--chunk-overlap", str(chunk_overlap), "--chunks-out", chunks_out]

    print(f"[CLEAN] Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    print("[CLEAN] Done.")

def run_index(in_clean_jsonl: str, rag_cfg_path: str) -> None:
    import yaml
    cfg = yaml.safe_load(Path(rag_cfg_path).read_text(encoding="utf-8")) or {}
    for key in ["chroma_dir", "embedding_model", "chunk_size", "chunk_overlap", "min_chars", "batch_limit"]:
        if key not in cfg:
            raise KeyError(f"[INDEX] Missing '{key}' in {rag_cfg_path}")
    Path(cfg["chroma_dir"]).mkdir(parents=True, exist_ok=True)

    ingest_jsonl_to_chroma(
        input_path=in_clean_jsonl,
        chroma_dir=cfg["chroma_dir"],
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
    cp.add_argument("--in-jsonl", required=True)
    cp.add_argument("--out-csv", default="data/processed/articles_clean.csv")
    cp.add_argument("--out-jsonl", default="data/processed/articles_clean.jsonl")
    cp.add_argument("--out-pre-jsonl", default="data/processed/preprocessed.jsonl")
    cp.add_argument("--tokenizer-model", default="facebook/bart-large-cnn")
    cp.add_argument("--truncate-to", type=int, default=1000)
    cp.add_argument("--min-chars", type=int, default=300)
    cp.add_argument("--min-words", type=int, default=80)
    cp.add_argument("--sentencer", choices=["nltk", "spacy"], default="nltk")
    cp.add_argument("--make-chunks", action="store_true")
    cp.add_argument("--chunk-tokens", type=int, default=800)
    cp.add_argument("--chunk-overlap", type=int, default=120)
    cp.add_argument("--chunks-out", default="data/chunks/chunks_{date}.jsonl")
    cp.add_argument("--date-from")
    cp.add_argument("--date-to")
    cp.add_argument("--sources", nargs="*")

    # index
    ip = sub.add_parser("index")
    ip.add_argument("--input", required=True, help="Clean JSONL (e.g., data/processed/articles_clean.jsonl)")
    ip.add_argument("--cfg", default="config/rag.yml")

    # all
    ap_all = sub.add_parser("all")
    ap_all.add_argument("--feeds", default="config/feeds.yml")
    ap_all.add_argument("--raw-jsonl", default="data/raw/articles.jsonl")
    ap_all.add_argument("--clean-csv", default="data/processed/articles_clean.csv")
    ap_all.add_argument("--clean-jsonl", default="data/processed/articles_clean.jsonl")
    ap_all.add_argument("--pre-jsonl", default="data/processed/preprocessed.jsonl")
    ap_all.add_argument("--rag-cfg", default="config/rag.yml")

    args = ap.parse_args()

    if args.cmd == "scrape":
        run_scrape(args.config, args.out_jsonl, per_request_sleep=args.sleep)

    elif args.cmd == "clean":
        run_clean(
            in_jsonl=args.in_jsonl,
            out_csv=args.out_csv,
            out_jsonl=args.out_jsonl,
            out_pre_jsonl=args.out_pre_jsonl,
            tokenizer_model=args.tokenizer_model,
            truncate_to=args.truncate_to,
            min_chars=args.min_chars,
            min_words=args.min_words,
            sentencer=args.sentencer,
            make_chunks=args.make_chunks,
            chunk_tokens=args.chunk_tokens,
            chunk_overlap=args.chunk_overlap,
            chunks_out=args.chunks_out,
            date_from=args.date_from,
            date_to=args.date_to,
            sources=args.sources,
        )

    elif args.cmd == "index":
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY is required for indexing.")
        run_index(args.input, args.cfg)

    elif args.cmd == "all":
        # 1) scrape
        run_scrape(args.feeds, args.raw_jsonl)
        # 2) clean (defaults)
        run_clean(
            in_jsonl=args.raw_jsonl,
            out_csv=args.clean_csv,
            out_jsonl=args.clean_jsonl,
            out_pre_jsonl=args.pre_jsonl,
            make_chunks=True,
        )
        # 3) index
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY is required for indexing.")
        run_index(args.clean_jsonl, args.rag_cfg)

if __name__ == "__main__":
    main()
