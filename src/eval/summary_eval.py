# eval/summary_eval.py
import json, argparse, numpy as np, pandas as pd
from tqdm import tqdm
from rouge_score import rouge_scorer
from bert_score import score as bertscore

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def bootstrap_ci(vals, n_boot=1000, alpha=0.05, seed=0):
    rng = np.random.default_rng(seed)
    v = np.array(vals, dtype=float)
    if len(v) == 0:
        return (0.0, 0.0, 0.0)
    boots = [rng.choice(v, size=len(v), replace=True).mean() for _ in range(n_boot)]
    lo, hi = np.quantile(boots, [alpha/2, 1-alpha/2])
    return float(v.mean()), float(lo), float(hi)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_jsonl", required=True)
    ap.add_argument("--refs_jsonl", default="data/gold/references.jsonl")
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--target_tokens", type=int, default=120)
    args = ap.parse_args()

    refs = {x["article_id"]: x for x in load_jsonl(args.refs_jsonl)}
    preds = [x for x in load_jsonl(args.pred_jsonl) if x["article_id"] in refs]

    rs = rouge_scorer.RougeScorer(["rouge1","rouge2","rougeL"], use_stemmer=True)

    r1 = []; r2 = []; rl = []
    cand_texts = []; ref_texts = []
    length_err = []; comp = []
    per_item = []

    for p in tqdm(preds, desc="scoring"):
        ref = refs[p["article_id"]]
        ref_sum = ref.get("reference_summary") or " ".join(ref["reference_bullets"])
        cand = p["summary"]
        sc = rs.score(ref_sum, cand)
        r1.append(sc["rouge1"].fmeasure)
        r2.append(sc["rouge2"].fmeasure)
        rl.append(sc["rougeL"].fmeasure)
        cand_texts.append(cand); ref_texts.append(ref_sum)

        length_err.append(abs(len(cand.split()) - args.target_tokens))
        if p.get("source_len"):
            comp.append(p["source_len"] / max(1, len(cand.split())))

        per_item.append({
            "article_id": p["article_id"],
            "rouge1_f": sc["rouge1"].fmeasure,
            "rouge2_f": sc["rouge2"].fmeasure,
            "rougeL_f": sc["rougeL"].fmeasure,
        })

    # BERTScore
    P, R, F = bertscore(cand_texts, ref_texts, lang="en", rescale_with_baseline=True)
    bs = list(F.numpy())

    metrics = {
        "rouge1_f": bootstrap_ci(r1),
        "rouge2_f": bootstrap_ci(r2),
        "rougeL_f": bootstrap_ci(rl),
        "bertscore_f1": bootstrap_ci(bs),
        "length_abs_error": bootstrap_ci(length_err),
        "compression_ratio": bootstrap_ci(comp) if comp else None,
        "n_items": (len(preds), None, None),
    }
    df = pd.DataFrame(
        [{"metric": k, "mean": v[0], "ci_lo": v[1], "ci_hi": v[2]} for k, v in metrics.items() if v]
    )
    df.to_csv(args.out_csv, index=False)
    print(df)
