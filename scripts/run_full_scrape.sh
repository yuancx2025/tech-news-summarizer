#!/usr/bin/env bash
set -euo pipefail

# Config
CFG="config/feeds.yml"
RAW="data/raw/articles.jsonl"
DB="data/processed/news.sqlite"
SLEEP="${SLEEP:-0.5}"   # allow override: SLEEP=0.7 ./scripts/run_full_scrape.sh
STAMP_FMT="${STAMP_FMT:-%Y-%m-%d}"  # or %Y-%m-%dT%H%MZ

# Ensure dirs exist
mkdir -p "$(dirname "$RAW")" "$(dirname "$DB")"

echo "[RUN] Scraping with $CFG ..."
python scripts/scrape_newspaper.py --config "$CFG" --sleep "$SLEEP"

ts="$(date +"$STAMP_FMT")"
RAW_SNAP="data/raw/articles_${ts}.jsonl"
DB_SNAP="data/processed/news_${ts}.sqlite"

echo "[SNAPSHOT] Copying $RAW -> $RAW_SNAP"
cp "$RAW" "$RAW_SNAP"

echo "[SNAPSHOT] Copying $DB -> $DB_SNAP"
cp "$DB" "$DB_SNAP"

echo "[DONE] Snapshot created:"
echo "  - $RAW_SNAP"
echo "  - $DB_SNAP"
