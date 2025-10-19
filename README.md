# RAG-Powered News Assistant 🚀📰

A full-stack AI-powered news platform that scrapes, analyzes, and provides intelligent summaries and Q&A for tech/finance news using **Retrieval-Augmented Generation (RAG)**, **semantic search**, and **analytics dashboards**.

## ✨ Key Features

| Feature | Description | Tech Stack |
|---------|-------------|------------|
| 🔍 **Semantic Search** | Vector-based article retrieval with MMR diversity | ChromaDB, OpenAI embeddings |
| 📝 **AI Summarization** | Multi-document map-reduce summarization | LangChain, Gemini 2.5 Flash |
| 🎯 **Smart Recommendations** | Personalized news feed with relevance + recency scoring | Custom ranking algorithm |
| 🤖 **Q&A Agent** | Timeline-based answers with natural language date parsing | dateparser, RAG pipeline |
| 📊 **Analytics Dashboard** | Real-time sentiment, ticker mentions, co-occurrence | Streamlit, Plotly, Pandas |
| 🌐 **REST API** | Full-featured backend with health checks & CORS | FastAPI, Pydantic |
| 📰 **Web Scraping** | Automated news collection from multiple sources | newspaper3k, feedparser |
| 💾 **Vector Database** | Persistent embeddings with metadata filtering | ChromaDB |
| 🧪 **Tested** | Comprehensive test suite for core functionality | pytest |

## 🚀 Quick Start

### Prerequisites

- Python 3.11+ (tested on 3.13)
- Google Gemini API key (for embeddings and LLM)
- 4GB+ RAM (for vector database)

### Installation

1. **Clone and navigate to the repository:**
   ```bash
   git clone https://github.com/yuancx2025/tech-news-summarizer.git
   cd tech-news-summarizer
   ```

2. **Create and activate virtual environment:**
   ```bash
   # Using venv
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

   # Or using conda (recommended)
   conda create -n ragnews python=3.13
   conda activate ragnews
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set environment variables:**
   ```bash
   export GOOGLE_API_KEY=<your-gemini-api-key>
   export RAG_CONFIG_PATH=config/rag.yml         # optional override
   export ANALYTICS_DIR=data/analytics            # optional override
   export FRONTEND_ORIGIN=http://localhost:8501   # optional CORS
   ```

   💡 **Tip**: Create a `.env` file in the project root:
   ```env
   GOOGLE_API_KEY=your_actual_key_here
   RAG_CONFIG_PATH=config/rag.yml
   ANALYTICS_DIR=data/analytics
   ```

## 📥 Data Pipeline: Building Your Knowledge Base

This project does **not** ship datasets or vector stores (to keep it lightweight and respect content licenses). You'll build everything locally using the automated pipeline.

### Pipeline Overview

```
Scrape → Clean → Index → Analytics
  ↓        ↓       ↓         ↓
 raw/   processed/ chroma/  metrics/
```

### Step 1: Configure News Sources

Edit [`config/feeds.yml`](config/feeds.yml) to customize which sites to scrape:

```yaml
sources:
  - domain: https://techcrunch.com
    limit: 30
    language: en
  - domain: https://www.theverge.com
    limit: 20
    language: en
  - domain: https://www.wired.com
    limit: 15
    language: en
```

### Step 2: Run End-to-End Pipeline (Recommended)

This command runs all stages in sequence:

```bash
export GOOGLE_API_KEY=<your-key>

python -m data_pipeline.pipeline all \
  --feeds config/feeds.yml \
  --rag-cfg config/rag.yml \
  --ticker-cfg config/tickers.yml
```

**What this does:**
1. **Scrapes** articles from configured sources → `data/raw/articles.jsonl`
2. **Cleans** text and validates metadata → `data/processed/articles_clean_valid_dates.jsonl`
3. **Indexes** into ChromaDB with embeddings → `data/vdb/chroma/`
4. **Generates** analytics (sentiment, mentions, co-occurrence) → `data/analytics/`

**Expected time:** ~10-15 minutes for 50-100 articles

---

### Step 3: Run Stages Individually (For Development)

If you need more control or already have partial data:

#### 3a. Scrape Raw Articles
```bash
python -m data_pipeline.pipeline scrape \
  --config config/feeds.yml \
  --out-jsonl data/raw/articles.jsonl
```

#### 3b. Clean & Validate
```bash
python -m data_pipeline.pipeline clean \
  --in-jsonl data/raw/articles.jsonl \
  --out-jsonl data/processed/articles_clean_valid_dates.jsonl \
  --out-csv data/processed/articles_clean.csv
```

#### 3c. Index into Vector Database
```bash
export GOOGLE_API_KEY=<your-key>

python -m data_pipeline.pipeline index \
  --input data/processed/articles_clean_valid_dates.jsonl \
  --cfg config/rag.yml
```
⚠️ **Important**: You MUST index articles before Q&A or recommendations work!

#### 3d. Generate Analytics Metrics
```bash
python -m data_pipeline.pipeline analytics \
  --rag-cfg config/rag.yml \
  --ticker-cfg config/tickers.yml \
  --out-dir data/analytics
```

---

### Step 4: Verify Data Integrity

Check that all pipeline outputs exist:

```bash
# Verify file structure
ls -lh data/raw/articles.jsonl
ls -lh data/processed/articles_clean_valid_dates.jsonl
ls -lh data/vdb/chroma/chroma.sqlite3
ls -lh data/analytics/overview.json
```

**Expected structure:**
```
data/
├── raw/
│   └── articles.jsonl                        # Raw scraped data
├── processed/
│   ├── articles_clean_valid_dates.jsonl     # Cleaned articles
│   └── validation_report.csv                 # Quality metrics
├── vdb/
│   └── chroma/
│       ├── chroma.sqlite3                    # Vector DB
│       └── <uuid>/                           # Embeddings
└── analytics/
    ├── overview.json                         # Dashboard data
    ├── ticker_cooccurrence.json              # Co-occurrence matrix
    ├── ticker_mentions.json                  # Mention counts
    ├── sector_sentiment.parquet              # Time-series sentiment
    └── sentiment.parquet                     # Cached scores
```

## 🎯 Running the Application

Once data is indexed, start the backend and frontend:

### Terminal 1: Start FastAPI Backend

```bash
export GOOGLE_API_KEY=<your-key>
export ANALYTICS_DIR=data/analytics  # optional

uvicorn src.api.api_main:app --host 0.0.0.0 --port 8000 --reload
```

**Test the API:**
```bash
# Health check
curl http://localhost:8000/health

# Expected: {"status":"ok","chroma_dir":"data/vdb/chroma"}
```

**Available endpoints:**
- `GET /health` - Health check
- `GET /meta/catalog` - Available tickers, sources, sections
- `POST /summarize` - Summarize article by ID/URL
- `POST /recommend` - Get personalized recommendations
- `POST /qa` - Ask questions (timeline-based Q&A)
- `GET /analytics/overview` - Sector sentiment & ticker mentions
- `GET /analytics/cooccurrence` - Ticker co-occurrence matrix

---

### Terminal 2: Start Main Streamlit UI

```bash
streamlit run app/ui_streamlit.py --server.port 8501
```

**Open in browser:** http://localhost:8501

**Features:**
- **📋 Summarize** tab: Get AI summaries of specific articles
- **💡 Recommend** tab: Personalized news feed based on interests
- **🤖 Ask** tab: Natural language Q&A with timeline view

---

### Terminal 3: Start Analytics Dashboard (Optional)

```bash
streamlit run app/dashboard_streamlit.py --server.port 8502
```

**Open in browser:** http://localhost:8502

**Features:**
- 📈 **Sector sentiment trends** (time-series chart)
- 📊 **Ticker mention counts** (bar chart)
- 🔗 **Co-occurrence heatmap** (which tickers appear together)
- 🎛️ **Interactive filters** (date range, ticker selection)

## Quick Reference

### Essential Commands

```bash
# 1. One-time setup
export GOOGLE_API_KEY=<your-key>
pip install -r requirements.txt

# 2. Build knowledge base (run once or when adding data)
python -m data_pipeline.pipeline all \
  --feeds config/feeds.yml \
  --rag-cfg config/rag.yml \
  --ticker-cfg config/tickers.yml

# 3. Start backend API
uvicorn src.api.api_main:app --host 0.0.0.0 --port 8000 --reload

# 4. Start main UI (in new terminal)
streamlit run app/ui_streamlit.py --server.port 8501

# 5. Start analytics dashboard (in new terminal, optional)
streamlit run app/dashboard_streamlit.py --server.port 8502
```

### API Quick Test

```bash
# Health check
curl http://localhost:8000/health

# Get available tickers/sources
curl http://localhost:8000/meta/catalog

# Ask a question
curl -X POST http://localhost:8000/qa \
  -H "Content-Type: application/json" \
  -d '{"question":"What is new?","days_back":90,"max_events":3}'

# Get analytics overview
curl "http://localhost:8000/analytics/overview?start=2024-01-01"
```

---

## 🤖 Question Answering Agent

The Q&A agent builds a **timeline of events** from relevant articles, then synthesizes a comprehensive answer with citations.

### How It Works

1. **Natural language date parsing** using `dateparser` (e.g., "last quarter", "since July")
2. **Semantic retrieval** with metadata filters (tickers, sources, dates)
3. **Timeline extraction** via LLM (Gemini) to identify key events
4. **Answer synthesis** that weaves events into a coherent narrative

### API Usage

**Endpoint:** `POST /qa`

**Request example:**
```bash
curl -X POST http://localhost:8000/qa \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are the latest developments in AI chips?",
    "temporal_filter": {
      "natural_language": "last 3 months",
      "days_back": 90
    },
    "tickers": ["NVDA", "AMD"],
    "sources": ["TechCrunch", "Wired"],
    "max_events": 5,
    "context_k": 16
  }'
```

**Response structure:**
```json
{
  "answer": "Recent AI chip developments include... (TechCrunch, 2024-10-15)",
  "timeline": [
    {
      "date": "2024-10-15",
      "summary": "NVIDIA announced next-gen Blackwell architecture",
      "source": "TechCrunch",
      "url": "https://...",
      "citation": "(TechCrunch, 2024-10-15)"
    },
    {
      "date": "2024-09-20",
      "summary": "AMD revealed MI300 series for data centers",
      "source": "Wired",
      "url": "https://...",
      "citation": "(Wired, 2024-09-20)"
    }
  ],
  "citations": ["(TechCrunch, 2024-10-15)", "(Wired, 2024-09-20)"]
}
```

### Streamlit "🤖 Ask" Tab

Interactive UI features:
- 💬 Free-text question input
- 📅 Natural language date hints ("last month", "since Q2 2024")
- 🏷️ Ticker/source filters (multi-select)
- ⚙️ Adjustable event count & retrieval depth
- 📊 Visual timeline with expandable event details

### Supported Temporal Queries

| Input | Interpretation |
|-------|---------------|
| "last 30 days" | Past 30 days from today |
| "since July" | July 1 to today |
| "Q3 2024" | July 1 - Sept 30, 2024 |
| "this quarter" | Current fiscal quarter |
| "yesterday" | Previous day |



## 🗂️ Project Structure

```
tech-news-summarizer/
├── app/                              # Streamlit frontends
│   ├── ui_streamlit.py              # Main UI (summarize/recommend/Q&A)
│   └── dashboard_streamlit.py       # Analytics dashboard
│
├── data_pipeline/                    # ETL pipeline
│   ├── __init__.py
│   ├── scrape.py                    # Web scraping (newspaper3k, feedparser)
│   ├── clean.py                     # Text normalization & validation
│   └── pipeline.py                  # Orchestrator (scrape → clean → index → analytics)
│
├── src/                              # Core backend
│   ├── embeddings.py                # OpenAI embedding utilities
│   ├── rag/                         # RAG components
│   │   ├── ingest.py               # ChromaDB ingestion
│   │   ├── retriever.py            # Semantic search & MMR
│   │   ├── ranking.py              # Relevance + recency scoring
│   │   ├── chains.py               # LangChain prompts & map-reduce
│   │   ├── schemas.py              # Pydantic models
│   │   ├── time_utils.py           # Natural language date parsing
│   │   └── tool.py                 # RAGTool (main orchestrator)
│   ├── analytics/                   # Analytics engine
│   │   ├── catalog.py              # Article collection from Chroma
│   │   ├── sentiment.py            # Sentiment scoring
│   │   └── metrics.py              # Metrics builder (sector, tickers, co-occurrence)
│   └── api/                         # FastAPI backend
│       └── api_main.py             # REST endpoints
│
├── config/                           # Configuration files
│   ├── feeds.yml                    # News sources & scraping params
│   ├── feeds_finance.yml            # Finance-specific sources
│   ├── rag.yml                      # RAG settings (embeddings, chunking, retrieval)
│   └── tickers.yml                  # Ticker symbol → sector mapping
│
├── data/                             # Data storage (gitignored)
│   ├── raw/                         # Scraped articles
│   ├── processed/                   # Cleaned & validated articles
│   ├── vdb/chroma/                  # ChromaDB vector store
│   └── analytics/                   # Pre-computed metrics (JSON/Parquet)
│
├── tests/                            # Test suite
│   ├── conftest.py                  # Pytest fixtures
│   ├── test_rag_qa.py              # RAG Q&A tests
│   ├── test_api_qa.py              # API endpoint tests
│   └── analytics/                   # Analytics module tests
│       ├── test_catalog.py
│       ├── test_metrics.py
│       └── test_sentiment.py
│
├── notebooks/                        # Jupyter notebooks for EDA
├── requirements.txt                  # Python dependencies
├── README.md
├── LICENSE
└── .gitignore
```

## 🧪 Testing

Run the test suite to verify functionality:

```bash
# Run all tests
pytest

# Run specific test modules
pytest tests/test_rag_qa.py -v
pytest tests/test_api_qa.py -v
pytest tests/analytics/ -v

# Run with coverage
pytest --cov=src --cov-report=html
```

**Test coverage includes:**
- RAG retrieval and ranking logic
- Q&A timeline extraction and synthesis
- Analytics metrics computation
- API endpoint validation
- Sentiment analysis accuracy

---

## 🔧 Configuration

### RAG Settings (`config/rag.yml`)

```yaml
chroma_dir: data/vdb/chroma
collection_name: articles
embedding:
  model: text-embedding-3-small
  provider: openai
  dimension: 1536
chunking:
  target_chunk_size: 800
  min_chunk_size: 400
  max_chunk_size: 1200
retrieval:
  k_final: 8
  fetch_k: 60
  lambda_mult: 0.6  # MMR diversity parameter
```

### Ticker Configuration (`config/tickers.yml`)

Map ticker symbols to sectors for analytics:

```yaml
tickers:
  AAPL:
    name: Apple Inc.
    sector: Technology
  NVDA:
    name: NVIDIA Corporation
    sector: Technology
  JPM:
    name: JPMorgan Chase
    sector: Finance
```

---

## 🚀 Deployment

### Production Checklist

- [ ] Set `GOOGLE_API_KEY` in production environment
- [ ] Configure rate limiting for API endpoints
- [ ] Set up Redis/Memcached for caching (optional)
- [ ] Enable CORS for your frontend domain
- [ ] Configure log aggregation (e.g., CloudWatch, Datadog)
- [ ] Set up monitoring alerts for API health
- [ ] Schedule periodic scraping (cron job or Airflow)

### Docker Deployment (Coming Soon)

```bash
# Build and run with Docker Compose
docker-compose up -d

# Services will be available at:
# - API: http://localhost:8000
# - UI: http://localhost:8501
# - Dashboard: http://localhost:8502
```