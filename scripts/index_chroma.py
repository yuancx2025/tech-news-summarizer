# scripts/index_chroma.py
import argparse, yaml
from src.rag.ingest import ingest_jsonl_to_chroma

def main():
    ap = argparse.ArgumentParser("Index cleaned news into Chroma")
    ap.add_argument("--in", dest="input_path", required=True, help="data/processed/articles_clean.jsonl")
    ap.add_argument("--cfg", dest="cfg_path", default="config/rag.yml")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.cfg_path, "r", encoding="utf-8"))
    ingest_jsonl_to_chroma(
        input_path=args.input_path,
        chroma_dir=cfg["chroma_dir"],
        embedding_model=cfg["embedding_model"],
        chunk_size=cfg["chunk_size"],
        chunk_overlap=cfg["chunk_overlap"],
        min_chars=cfg["min_chars"],
        separators=cfg.get("separators"),
        batch_limit=cfg["batch_limit"],
    )

if __name__ == "__main__":
    main()
