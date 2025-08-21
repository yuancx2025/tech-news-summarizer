import json, argparse, pathlib, time
from typing import List, Dict
from tqdm import tqdm
import evaluate  # ROUGE
try:
    bertscore = evaluate.load("bertscore")
except Exception:
    bertscore = None
rouge = evaluate.load("rouge")

from model.summarizers import build_summarizers

def load_jsonl(path: str) -> List[Dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-jsonl", required=True, help="JSONL with id, article, reference (ref may be empty)")
    ap.add_argument("--models", default="lead3,textrank,distilbart,bart")
    ap.add_argument("--outdir", default="results/baselines")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", type=int, default=None, help="GPU index for HF (None/-1=CPU)")
    ap.add_argument("--min-length", type=int, default=60)
    ap.add_argument("--max-length", type=int, default=200)
    ap.add_argument("--beams", type=int, default=4)
    args = ap.parse_args()

    outdir = pathlib.Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    data = load_jsonl(args.test_jsonl)
    if args.limit > 0: data = data[: args.limit]

    # Build models (pass decoding config)
    summarizers = build_summarizers()
    # Update HF configs if provided
    if "distilbart" in summarizers.__dict__ or "bart" in summarizers.__dict__:
        pass  # We created with defaults; if you want to wire args, modify HFSummarizer init

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    summaries = {m: [] for m in models}
    references = []
    ids = []

    # Generate
    print(f"Running models: {models} on {len(data)} articles")
    t0 = time.time()
    for row in tqdm(data, desc="Summarizing"):
        text = row.get("article") or ""
        ref = row.get("reference", "")
        references.append(ref)
        ids.append(row["id"])
        for m in models:
            pred = summarizers[m].summarize(text)
            summaries[m].append(pred)
    t1 = time.time()

    # Save predictions
    for m in models:
        with open(outdir / f"preds_{m}.jsonl", "w", encoding="utf-8") as f:
            for id_, pred, ref in zip(ids, summaries[m], references):
                f.write(json.dumps({"id": id_, "prediction": pred, "reference": ref}, ensure_ascii=False) + "\n")

    # Evaluate if references exist
    nonempty = [i for i, r in enumerate(references) if r and str(r).strip()]
    summary_rows = []
    if nonempty:
        refs = [references[i] for i in nonempty]
        for m in models:
            preds = [summaries[m][i] for i in nonempty]
            r = rouge.compute(predictions=preds, references=refs, use_stemmer=True)
            metrics = {"model": m, **r}
            if bertscore:
                b = bertscore.compute(predictions=preds, references=refs, lang="en")
                metrics.update({
                    "bertscore_precision": sum(b["precision"])/len(b["precision"]),
                    "bertscore_recall":    sum(b["recall"])/len(b["recall"]),
                    "bertscore_f1":        sum(b["f1"])/len(b["f1"]),
                })
            with open(outdir / f"metrics_{m}.json", "w", encoding="utf-8") as f:
                f.write(json.dumps(metrics, indent=2))
            summary_rows.append(metrics)
    else:
        print("[note] No references provided; skipping metrics. Use preds_*.jsonl for qualitative review.")

    # Write a compact CSV summary if any metrics available
    if summary_rows:
        import pandas as pd
        pd.DataFrame(summary_rows).to_csv(outdir / "summary.csv", index=False)

    # Throughput info
    print(f"Total time: {(t1 - t0):.2f}s for {len(data)} articles")

if __name__ == "__main__":
    main()
