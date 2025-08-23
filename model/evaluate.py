"""
model/evaluate.py

Evaluation script for summarization models.
One script to:
- load preprocessed JSONL
- build evaluation set (optionally merging references from CSV or a field in JSONL)
- run baselines (Lead-3, TextRank, DistilBART, BART)
- compute simple metrics
- save predictions and metrics to results/<run_name>/

Usage examples:
  # Basic evaluation (qualitative)
  python -m model.evaluate \
    --pre-jsonl data/processed/preprocessed_2025-08-19.jsonl \
    --run-name qual_check \
    --models lead3,textrank \
    --limit 100

  # Evaluation with references
  python -m model.evaluate \
    --pre-jsonl data/processed/preprocessed_2025-08-19.jsonl \
    --ref-field summary \
    --run-name with_refs \
    --models lead3,textrank \
    --limit 100
"""

from __future__ import annotations
import argparse, csv, json, pathlib, time, sys
from typing import Dict, List, Optional

# Import summarizers from the same package
from .summarizers import build_summarizers

# --- IO helpers ---
def load_preprocessed_jsonl(path: str) -> List[Dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows

def load_refs_csv(path: str) -> Dict[str, str]:
    refs = {}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            refs[str(r["id"])] = r["reference"]
    return refs

def pick_article(row: Dict) -> str:
    return row.get("model_input") or row.get("full_text") or ""

def pick_sentences(row: Dict) -> Optional[List[str]]:
    s = row.get("sentences")
    return s if isinstance(s, list) else None

# --- Build references in-memory ---
def build_references(data: List[Dict], refs_csv: Optional[str], ref_field: Optional[str], pseudo_ref: Optional[str]) -> List[str]:
    """Return list of reference strings ('' when not available)."""
    # Priority: CSV > ref_field in JSONL > pseudo_ref > empty
    refs_map = {}
    if refs_csv:
        refs_map = load_refs_csv(refs_csv)

    refs_out = []
    for row in data:
        rid = str(row.get("id", ""))
        ref = ""
        if refs_map:
            ref = refs_map.get(rid, "")
        if (not ref) and ref_field:
            # Accept a field directly from JSONL (e.g., 'summary' or 'standfirst')
            val = row.get(ref_field)
            if isinstance(val, (list, tuple)):
                ref = " ".join([str(x) for x in val])
            elif isinstance(val, str):
                ref = val
        if (not ref) and pseudo_ref:
            # Pragmatic pseudo-gold for plumbing tests (NOT a real evaluation)
            if pseudo_ref == "lead3":
                sents = pick_sentences(row)
                if sents:
                    ref = " ".join(sents[:3]).strip()
                else:
                    import re
                    sents = re.split(r'(?<=[.!?])\s+', pick_article(row))
                    ref = " ".join(sents[:3]).strip()
        refs_out.append(ref or "")
    return refs_out

def compute_simple_metrics(predictions: List[str], references: List[str]) -> Dict:
    """Compute simple word overlap metrics."""
    metrics = {}
    
    # Word overlap ratio
    total_overlap = 0
    total_words = 0
    for pred, ref in zip(predictions, references):
        if not ref.strip():
            continue
        pred_words = set(pred.lower().split())
        ref_words = set(ref.lower().split())
        if ref_words:
            overlap = len(pred_words & ref_words)
            total_overlap += overlap
            total_words += len(ref_words)
    
    if total_words > 0:
        metrics["word_overlap_ratio"] = total_overlap / total_words
    else:
        metrics["word_overlap_ratio"] = 0.0
    
    # Average summary length
    pred_lengths = [len(pred.split()) for pred in predictions if pred.strip()]
    if pred_lengths:
        metrics["avg_summary_length"] = sum(pred_lengths) / len(pred_lengths)
    else:
        metrics["avg_summary_length"] = 0
    
    return metrics

def main():
    print("Starting baseline evaluation...")
    ap = argparse.ArgumentParser()
    ap.add_argument("--pre-jsonl", required=True, help="Preprocessed JSONL (from clean_export_preprocess.py)")
    ap.add_argument("--refs-csv", default=None, help="Optional CSV mapping id->reference (columns: id,reference)")
    ap.add_argument("--ref-field", default=None, help="Optional field name in JSONL to use as reference (e.g., 'summary')")
    ap.add_argument("--pseudo-ref", choices=["lead3"], default=None, help="Generate a pseudo reference (for plumbing tests)")
    ap.add_argument("--models", default="lead3,textrank,distilbart,bart")
    ap.add_argument("--run-name", default="run")
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    print(f"Arguments parsed: {args}")

    outdir = pathlib.Path(args.outdir) / args.run_name
    outdir.mkdir(parents=True, exist_ok=True)

    data = load_preprocessed_jsonl(args.pre_jsonl)
    if args.limit > 0:
        data = data[: args.limit]
    print(f"[info] Loaded {len(data)} examples from {args.pre_jsonl}")

    # Build references (in-memory)
    references = build_references(data, args.refs_csv, args.ref_field, args.pseudo_ref)
    have_refs = any(bool(r and str(r).strip()) for r in references)
    if args.refs_csv and not have_refs:
        print("[warn] --refs-csv provided but no matching ids found; proceeding without refs.")
    if args.ref_field and not have_refs:
        print(f"[warn] --ref-field '{args.ref_field}' found no non-empty values; proceeding without refs.")
    if args.pseudo_ref and not have_refs:
        print("[warn] --pseudo-ref requested but produced no refs; proceeding without refs.")

    # Build summarizers
    print("[info] Building summarizers...")
    summarizers = build_summarizers()
    print(f"[info] Built summarizers: {list(summarizers.keys())}")

    model_names = [m.strip() for m in args.models.split(",") if m.strip()]
    summaries = {m: [] for m in model_names}
    ids = [str(row.get("id","")) for row in data]

    # Generate predictions
    t0 = time.time()
    for i, row in enumerate(data):
        if i % 10 == 0:
            print(f"[info] Processing example {i+1}/{len(data)}")
        text = pick_article(row)
        sents = pick_sentences(row)
        for m in model_names:
            try:
                pred = summarizers[m].summarize(text, sentences=sents)
                summaries[m].append(pred)
            except Exception as e:
                print(f"[error] Failed to generate summary for model {m}, example {i}: {e}")
                summaries[m].append("")  # Add empty string as fallback
    t1 = time.time()
    print(f"[info] Generated predictions for {len(data)} examples in {(t1 - t0):.2f}s")

    # Save per-model predictions
    for m in model_names:
        with open(outdir / f"preds_{m}.jsonl", "w", encoding="utf-8") as f:
            for id_, pred, ref in zip(ids, summaries[m], references):
                f.write(json.dumps({"id": id_, "prediction": pred, "reference": ref}, ensure_ascii=False) + "\n")
        print(f"[ok] wrote {outdir / f'preds_{m}.jsonl'}")

    # Compute and save metrics
    nonempty = [i for i, r in enumerate(references) if r and str(r).strip()]
    summary_rows = []
    
    if nonempty:
        print(f"[info] Computing metrics for {len(nonempty)} examples with references")
        refs = [references[i] for i in nonempty]
        
        for m in model_names:
            preds = [summaries[m][i] for i in nonempty]
            metrics = {"model": m}
            
            # Add simple metrics
            simple_metrics = compute_simple_metrics(preds, refs)
            metrics.update(simple_metrics)
            
            with open(outdir / f"metrics_{m}.json", "w", encoding="utf-8") as f:
                f.write(json.dumps(metrics, indent=2))
            summary_rows.append(metrics)
            print(f"[ok] wrote {outdir / f'metrics_{m}.json'}")
        
        # Write compact CSV summary
        if summary_rows:
            try:
                import pandas as pd
                pd.DataFrame(summary_rows).to_csv(outdir / "summary.csv", index=False)
                print(f"[ok] wrote {outdir / 'summary.csv'}")
            except Exception as e:
                print(f"[warn] could not write summary.csv: {e}")
    else:
        print("[note] No references available → skipping metric computation. Use preds_*.jsonl for qualitative review.")

    print("[done]")

if __name__ == "__main__":
    main()
