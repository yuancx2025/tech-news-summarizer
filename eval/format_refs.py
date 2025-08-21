import json, csv, argparse, pathlib

def load_preprocessed(path):
    rows = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                x = json.loads(line)
                # prefer model_input (truncated), else full_text
                rows[str(x["id"])] = x.get("model_input") or x.get("full_text") or ""
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pre-jsonl", required=True)
    ap.add_argument("--refs-csv", default=None)  # CSV: id,reference
    ap.add_argument("--out-jsonl", required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    articles = load_preprocessed(args.pre_jsonl)
    refs = {}
    if args.refs_csv:
        with open(args.refs_csv, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                refs[str(r["id"])] = r["reference"]

    outp = pathlib.Path(args.out_jsonl); outp.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(outp, "w", encoding="utf-8") as out:
        for id_, art in articles.items():
            ref = refs.get(id_, "")
            out.write(json.dumps({"id": id_, "article": art, "reference": ref}, ensure_ascii=False) + "\n")
            n += 1
            if args.limit and n >= args.limit: break
    print(f"Wrote {n} rows → {outp}")

if __name__ == "__main__":
    main()
