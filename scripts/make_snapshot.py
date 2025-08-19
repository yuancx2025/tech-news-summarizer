import json, hashlib, argparse, pathlib, sqlite3, statistics, subprocess
from datetime import datetime, timezone
from dateutil import parser as dateparser
from typing import Dict, Any, Iterable, Optional
from tqdm import tqdm

# Optional tokenizer (loaded lazily)
_TOK = None

def load_tokenizer(model_name: Optional[str]):
    global _TOK
    if model_name:
        from transformers import AutoTokenizer
        _TOK = AutoTokenizer.from_pretrained(model_name)
    return _TOK

def count_tokens(text: str) -> Optional[int]:
    if _TOK is None:
        return None
    return len(_TOK(text, add_special_tokens=True, truncation=False)["input_ids"])

def truncate_to_tokens(text: str, max_tokens: int) -> str:
    enc = _TOK(text, max_length=max_tokens, truncation=True)
    return _TOK.decode(enc["input_ids"], skip_special_tokens=True)

def iso8601(dt_str: str) -> str:
    dt = dateparser.parse(dt_str)
    if dt is None:
        raise ValueError("unparseable date")
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")

def normalize_source(url: str, fallback: str = "unknown") -> str:
    try:
        from urllib.parse import urlparse
        netloc = urlparse(url).netloc.lower()
        src = netloc.replace("www.", "") or fallback
        # Optional aliasing hook:
        # You can map subdomains to canonical names here if needed.
        return src
    except Exception:
        return fallback

def stable_id(url: str, title: str, date_iso: str) -> str:
    h = hashlib.sha1()
    h.update((url.strip() + "|" + title.strip() + "|" + date_iso).encode("utf-8"))
    return h.hexdigest()

def text_fingerprint(s: str) -> str:
    clean = " ".join(s.split()).lower()
    return hashlib.md5(clean.encode("utf-8")).hexdigest()

def iter_jsonl(glob_pat: str) -> Iterable[Dict[str, Any]]:
    for p in sorted(pathlib.Path().glob(glob_pat)):
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)

def iter_sqlite(path: str, table: str = "articles", where: Optional[str] = None) -> Iterable[Dict[str, Any]]:
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    cur = con.cursor()
    sql = f"SELECT * FROM {table}"
    if where:
        sql += " WHERE " + where
    for row in cur.execute(sql):
        yield dict(row)
    con.close()

def get_git_commit() -> Optional[str]:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None

def build_snapshot(
    input_mode: str,
    input_arg: str,
    sqlite_table: str,
    sqlite_where: Optional[str],
    out_jsonl: pathlib.Path,
    manifest_path: pathlib.Path,
    min_chars: int,
    tokenizer_model: Optional[str],
    truncate_to: Optional[int],
    per_source_cap: Optional[int],
):
    load_tokenizer(tokenizer_model)

    seen_fp = set()
    kept = 0
    dropped_too_short = 0
    dropped_dupe = 0
    dropped_malformed = 0
    token_counts = []
    sources: Dict[str, int] = {}

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    loader = (
        iter_jsonl(input_arg)
        if input_mode == "jsonl"
        else iter_sqlite(input_arg, sqlite_table, sqlite_where)
    )

    with out_jsonl.open("w", encoding="utf-8") as out:
        for raw in tqdm(loader, desc="Snapshotting"):
            title = (raw.get("title") or "").strip()
            url = (raw.get("url") or "").strip()
            full_text = (raw.get("full_text") or raw.get("text") or "").strip()
            raw_date = raw.get("date") or raw.get("published_at") or raw.get("time")

            # 1) minimal validation
            if not (title and url and full_text and raw_date):
                dropped_malformed += 1
                continue
            if len(full_text) < min_chars:
                dropped_too_short += 1
                continue
            try:
                date_iso = iso8601(str(raw_date))
            except Exception:
                dropped_malformed += 1
                continue

            # 2) normalize source
            source = (raw.get("source") or normalize_source(url)).strip() or "unknown"

            # 3) per-source cap (optional)
            if per_source_cap is not None:
                if sources.get(source, 0) >= per_source_cap:
                    continue  # silently cap; you can also count caps if preferred

            # 4) dedupe
            fp = text_fingerprint(full_text + "|" + url)
            if fp in seen_fp:
                dropped_dupe += 1
                continue
            seen_fp.add(fp)

            # 5) tokenizer stats + truncation
            model_input = None
            tok_count = count_tokens(full_text)
            if tok_count is not None:
                token_counts.append(tok_count)
                if truncate_to and truncate_to > 0:
                    model_input = truncate_to_tokens(full_text, truncate_to)

            # 6) stable id (prefer upstream id if present)
            aid = str(raw.get("id") or stable_id(url, title, date_iso))

            rec = {
                "id": aid,
                "title": title,
                "source": source,
                "date": date_iso,
                "url": url,
                "full_text": full_text,
            }
            if "category" in raw and raw.get("category") not in (None, ""):
                rec["category"] = raw.get("category")
            if "tags" in raw and raw.get("tags"):
                rec["tags"] = raw.get("tags")
            if model_input:
                rec["model_input"] = model_input

            sources[source] = sources.get(source, 0) + 1
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            kept += 1

    # 7) manifest
    manifest = {
        "snapshot_path": str(out_jsonl),
        "created_at": datetime.utcnow().isoformat() + "Z",
        "git_commit": get_git_commit(),
        "rows_kept": kept,
        "dropped_too_short": dropped_too_short,
        "dropped_duplicate": dropped_dupe,
        "dropped_malformed": dropped_malformed,
        "sources_top": sorted([[k, v] for k, v in sources.items()], key=lambda x: -x[1])[:25],
        "tokenizer": tokenizer_model or None,
        "truncate_to_tokens": truncate_to or None,
        "per_source_cap": per_source_cap,
        "snapshot_version": "1.0.0",
    }

    if token_counts:
        token_counts_sorted = sorted(token_counts)
        def pct(p):
            k = max(0, min(len(token_counts_sorted)-1, int(round((p/100.0)*(len(token_counts_sorted)-1)))))
            return token_counts_sorted[k]
        manifest["token_stats"] = {
            "counted": len(token_counts),
            "mean": float(statistics.fmean(token_counts)),
            "p50": float(pct(50)),
            "p90": float(pct(90)),
            "p95": float(pct(95)),
            "p99": float(pct(99)),
            "over_1024_pct": float(sum(t > 1024 for t in token_counts)) / len(token_counts),
        }

    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["jsonl", "sqlite"], default="jsonl")
    ap.add_argument("--input", required=True, help="glob for JSONL or path to SQLite")
    ap.add_argument("--sqlite-table", default="articles")
    ap.add_argument("--sqlite-where", default=None, help='e.g., "date >= \'2025-07-01\'"')
    ap.add_argument("--out", default=None)
    ap.add_argument("--min-chars", type=int, default=300)
    ap.add_argument("--tokenizer-model", default=None, help="e.g., facebook/bart-large-cnn")
    ap.add_argument("--truncate-to", type=int, default=0, help="truncate to N tokens for model_input")
    ap.add_argument("--per-source-cap", type=int, default=None, help="max rows per source (optional)")
    args = ap.parse_args()

    stamp = datetime.utcnow().strftime("%Y-%m-%d")
    out = args.out or f"data/processed/news_baseline_{stamp}.jsonl"
    manifest = out.replace(".jsonl", "_manifest.json")

    manifest_dict = build_snapshot(
        input_mode=args.mode,
        input_arg=args.input,
        sqlite_table=args.sqlite_table,
        sqlite_where=args.sqlite_where,
        out_jsonl=pathlib.Path(out),
        manifest_path=pathlib.Path(manifest),
        min_chars=args.min_chars,
        tokenizer_model=args.tokenizer_model,
        truncate_to=(args.truncate_to if args.truncate_to > 0 else None),
        per_source_cap=args.per_source_cap,
    )
    print(json.dumps(manifest_dict, indent=2))

if __name__ == "__main__":
    main()
