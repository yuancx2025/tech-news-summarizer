# eval/prepare_cnndm_gold.py
import argparse, json, pathlib
from datasets import load_dataset

def to_bullets(highlights: str):
    # CNNDM "highlights" are newline-separated bullets in most cases.
    # Fall back to sentence-ish splits if needed.
    lines = [x.strip(" •-") for x in highlights.split("\n") if x.strip()]
    if lines:
        return lines
    # fallback
    import re
    sents = re.split(r"(?<=[.!?])\s+", highlights.strip())
    return [s for s in sents if s]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="validation", choices=["train","validation","test"])
    ap.add_argument("--n", type=int, default=300, help="how many examples to export")
    ap.add_argument("--outdir", default="data/gold")
    args = ap.parse_args()

    outdir = pathlib.Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    ds = load_dataset("cnn_dailymail", "3.0.0", split=args.split, trust_remote_code=True)
    ds = ds.select(range(min(args.n, len(ds))))

    refs_fp = (outdir / "references.jsonl").open("w", encoding="utf-8")
    kf_fp   = (outdir / "key_facts.jsonl").open("w", encoding="utf-8")
    art_fp  = (outdir / "articles.jsonl").open("w", encoding="utf-8")

    for ex in ds:
        article_id = f"cnn_{ex['id']}"
        bullets = to_bullets(ex["highlights"])
        ref_sum = " ".join(bullets) if bullets else ex["highlights"].strip()
        refs_fp.write(json.dumps({
            "article_id": article_id,
            "reference_bullets": bullets,
            "reference_summary": ref_sum
        }, ensure_ascii=False) + "\n")
        kf_fp.write(json.dumps({
            "article_id": article_id,
            "key_facts": bullets  # treat highlights as key facts for coverage later
        }, ensure_ascii=False) + "\n")
        art_fp.write(json.dumps({
            "article_id": article_id,
            "article": ex["article"]
        }, ensure_ascii=False) + "\n")

    for fp in (refs_fp, kf_fp, art_fp): fp.close()

if __name__ == "__main__":
    main()
