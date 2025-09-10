# LLM-Powered Tech News Summarizer 🚀📰

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
  --rag-cfg config/rag.yml
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

