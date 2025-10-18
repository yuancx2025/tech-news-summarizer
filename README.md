# RAG-Powered News Assistant 🚀📰

This project builds a full-stack pipeline to scrape, clean, summarize, and recommend tech/finance news using Retrieval-Augmented Generation (RAG).

## 🚀 Quick Start

### Installation

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd tech-news-summarizer
   ```

2. **Create and activate virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## 📥 Rebuilding Scraped Data

This project does **not** ship full datasets or vector stores in the repo (to keep it lightweight and respect site content licenses).  
Instead, you can reproduce everything locally with the provided data pipeline.

### 1. Configure Sources
Edit [`config/feeds.yml`](config/feeds.yml) to specify which news sites to scrape:

```yaml
sources:
  - domain: https://techcrunch.com
    limit: 20
    language: en
  - domain: https://www.theverge.com
    limit: 10
    language: en
```

### 2. Run the pipeline (end-to-end)
From the repo root:

```
# Scrape → Clean → Index into Chroma
python -m data_pipeline.pipeline all \
  --feeds config/feeds.yml \
  --rag-cfg config/rag.yml \
  --ticker-cfg config/tickers.yml
```
This will generate:

- Raw scraped data → data/raw/articles.jsonl
- Cleaned & preprocessed data → data/processed/articles_clean.csv/jsonl, data/processed/preprocessed.jsonl
- Vector database (Chroma) → data/vdb/chroma/
- Analytics artifacts (JSON/Parquet) → data/analytics/

### 3. Run Stages Individually (Optional)

```
# Scrape raw articles
python -m data_pipeline.pipeline scrape \
  --config config/feeds.yml \
  --out-jsonl data/raw/articles.jsonl

# Clean + preprocess
python -m data_pipeline.pipeline clean \
  --in-jsonl data/raw/articles.jsonl \
  --out-csv data/processed/articles_clean.csv \
  --out-jsonl data/processed/articles_clean.jsonl \
  --out-pre-jsonl data/processed/preprocessed.jsonl \
  --make-chunks

# Index into Chroma
python -m data_pipeline.pipeline index \
  --input data/processed/articles_clean.jsonl \
  --cfg config/rag.yml

# Analytics metrics + dashboards
python -m data_pipeline.pipeline analytics \
  --rag-cfg config/rag.yml \
  --ticker-cfg config/tickers.yml \
  --out-dir data/analytics
```

### 4. Verify Outputs

- Raw data: data/raw/
- Cleaned data: data/processed/
- Vector DB: data/vdb/chroma/
- Analytics cache & metrics: data/analytics/

## 📈 Analytics & Dashboard

1. **Regenerate metrics** (after indexing or when new articles arrive):
   ```bash
   python -m data_pipeline.pipeline analytics \
     --rag-cfg config/rag.yml \
     --ticker-cfg config/tickers.yml \
     --out-dir data/analytics
   ```
   This will
   - Extract article metadata from Chroma with ticker/sector enrichment.
   - Score sentiment per article (cached at `data/analytics/sentiment.parquet`).
   - Persist sector sentiment, ticker mentions, and co-occurrence matrices as JSON + Parquet under `data/analytics/`.

2. **Expose analytics via FastAPI**: set `ANALYTICS_DIR` if you keep metrics outside the default.
   ```bash
   export ANALYTICS_DIR=/path/to/data/analytics
   uvicorn src.api.api_main:app --host 0.0.0.0 --port 8000 --reload
   ```

3. **Launch the Streamlit analytics dashboard**:
   ```bash
   streamlit run app/dashboard_streamlit.py --server.port 8502
   ```
   Use the sidebar controls to adjust date/ticker filters. The charts call the FastAPI analytics endpoints (`/analytics/overview`, `/analytics/cooccurrence`).

## 🚀 How to run FastAPI + Streamlit with Chroma:

### 1. Start the FastAPI Backend
```
# open a terminal and run
uvicorn src.api.api_main:app --host 0.0.0.0 --port 8000 --reload
```

### 2. Start the Streamlit Frontend
```
# Open a second terminal and run:
streamlit run app/ui_streamlit.py --server.port 8501
```

## 📌 Project Goals
- Scrape news from sources like TechCrunch, Wired, and The Verge
- Summarize and making recommendation with RAG
- Deploy via Streamlit with interactive UI
- Visualize insights using Tableau

## 🗂️ Project Structure
```text
tech-news-summarizer/
├── app/                      # Frontend/UI
│   └── ui_streamlit.py
│
├── data_pipeline/            # Data prep & indexing
│   ├── __init__.py
│   ├── scrape.py             # newspaper3k scraping
│   ├── clean.py              # cleaning & normalization
│   ├── embed.py              # embeddings + OpenAI API
│   ├── index_chroma.py       # ingest into Chroma/FAISS
│   └── pipeline.py           # orchestrate full pipeline
│
├── src/                      # Core backend logic
│   ├── __init__.py
│   ├── rag/                  # Retrieval + RAG
│   │   ├── __init__.py
│   │   ├── retriever.py
│   │   ├── ranker.py
│   │   ├── rag_summarizer.py
│   │   └── recommend.py
│   ├── api/                  # API layer
│   │   ├── __init__.py
│   │   └── main.py           # FastAPI app
│   └── schemas.py            # Pydantic models shared across RAG/API
│
├── config/                   # YAML configs
│   ├── feeds.yml
│   └── rag.yml
│
├── data/                     # Local storage (gitignored)
│   ├── raw/
│   ├── processed/
│   ├── chunks/
│   └── vdb/
│
├── tests/                    # Unit + E2E tests
├── notebooks/                # EDA & experiments
├── cli.py                    # optional: Typer CLI as single entrypoint
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore

```

## 🚧 Project Milestones

| Week | Milestone |
|------|-----------|
| ✅ Week 1 | Scraped and cleaned tech articles |
| ✅ Week 2 | Built and evaluated LLM summarizer |
| ✅ Week 3 | RAG-based improvement with Chroma |
| ✅ Week 4 | RAG-based recommendation |
| ✅ Week 5 | Streamlit app deployment + demo |

