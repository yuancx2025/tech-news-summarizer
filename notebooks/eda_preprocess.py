# notebooks/eda_preprocessed.py
import json, pathlib, warnings
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer

SNAPSHOT = "data/processed/preprocessed_YYYY-MM-DD.jsonl"  # <- set to your file
REPORT_DIR = pathlib.Path("reports"); REPORT_DIR.mkdir(parents=True, exist_ok=True)

pd.set_option("display.max_colwidth", 140)
warnings.filterwarnings("ignore")

def load_jsonl(path: str) -> pd.DataFrame:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return pd.DataFrame(rows)

df = load_jsonl(SNAPSHOT)
print(f"Loaded {len(df):,} rows")
assert {"id","title","source","date","url","full_text"}.issubset(df.columns)

# --- Coverage
rows = len(df)
n_sources = df["source"].nunique()
date_min, date_max = df["date"].min(), df["date"].max()
print(f"Rows: {rows:,} | Sources: {n_sources} | Range: {date_min} → {date_max}")

# --- Sources
top_sources = df["source"].value_counts().head(20)
top_sources.to_csv(REPORT_DIR/"eda_top_sources.csv")
plt.figure()
top_sources.plot(kind="bar")
plt.title("Top sources by count")
plt.ylabel("articles")
plt.tight_layout()
plt.savefig(REPORT_DIR/"eda_source_bar.png", dpi=160)
plt.close()

# --- Time series
df["date_day"] = pd.to_datetime(df["date"]).dt.date
ts = df.groupby("date_day").size()
ts.to_csv(REPORT_DIR/"eda_timeseries.csv")
plt.figure()
ts.plot()
plt.title("Articles per day")
plt.xlabel("date"); plt.ylabel("count")
plt.tight_layout()
plt.savefig(REPORT_DIR/"eda_timeseries.png", dpi=160)
plt.close()

# --- Lengths: words & tokens
df["words_full"] = df["full_text"].astype(str).str.split().map(len)
print(df[["words_full"]].describe())

token_cols = [c for c in ["tokens_full","tokens_input"] if c in df.columns]
for col in token_cols:
    desc = df[col].dropna().describe()
    print(f"\n{col} stats:\n{desc}")
    plt.figure()
    df[col].dropna().hist(bins=60)
    plt.title(f"{col} distribution")
    plt.xlabel("tokens"); plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(REPORT_DIR/f"eda_{col}_hist.png", dpi=160)
    plt.close()

if "tokens_full" in df.columns and "tokens_input" in df.columns:
    over_1024 = (df["tokens_full"] > 1024).mean()
    compress = (df["tokens_input"] / df["tokens_full"]).replace([np.inf, -np.inf], np.nan).dropna()
    print(f"% full_text over 1024 tokens: {over_1024:.1%}")
    print(f"Compression ratio (input/full): median={compress.median():.3f}, p90={compress.quantile(0.9):.3f}")

# --- Sentences availability (for Lead-3)
has_sents = "sentences" in df.columns
if has_sents:
    df["num_sents"] = df["sentences"].map(lambda s: len(s) if isinstance(s, list) else 0)
    print(df["num_sents"].describe())
    too_short = (df["num_sents"] < 3).mean()
    print(f"% with <3 sentences (Lead-3 not ideal): {too_short:.1%}")

# --- Quick n-grams (on model_input if present, else full_text)
text_series = df["model_input"].fillna(df["full_text"]).astype(str)
cv = CountVectorizer(stop_words="english", max_features=20000, ngram_range=(1,2))
X = cv.fit_transform(text_series)
vocab = np.array(cv.get_feature_names_out())
freqs = np.asarray(X.sum(axis=0)).ravel()
idx = freqs.argsort()[::-1][:100]
ngrams = pd.DataFrame({"ngram": vocab[idx], "freq": freqs[idx]})
ngrams.to_csv(REPORT_DIR/"eda_top_ngrams.csv", index=False)

# --- KPI summary row
summary = {
    "rows": int(rows),
    "unique_sources": int(n_sources),
    "date_min": str(date_min),
    "date_max": str(date_max),
    "median_words_full": float(df["words_full"].median()),
    "p95_words_full": float(df["words_full"].quantile(0.95)),
}
if "tokens_full" in df.columns:
    summary |= {
        "mean_tokens_full": float(df["tokens_full"].dropna().mean()),
        "p95_tokens_full": float(df["tokens_full"].dropna().quantile(0.95)),
        "pct_full_over_1024": float((df["tokens_full"] > 1024).mean()),
    }
if "tokens_input" in df.columns:
    summary |= {
        "mean_tokens_input": float(df["tokens_input"].dropna().mean()),
        "p95_tokens_input": float(df["tokens_input"].dropna().quantile(0.95)),
    }
pd.DataFrame([summary]).to_csv(REPORT_DIR/"eda_summary.csv", index=False)
print("Saved reports to ./reports/")
