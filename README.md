# LLM-Powered Tech News Summarizer 🚀📰

This project builds a full-stack pipeline to scrape, clean, summarize, and visualize tech news using large language models like BART and T5.

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

4. **Installation complete!** The TextRank summarizer now uses a custom sentence tokenizer that doesn't require additional downloads.

## 🧪 Running Evaluations

After installation, you can evaluate different summarization models:

### Basic Evaluation (Qualitative)
```bash
python -m model.evaluate \
  --pre-jsonl data/processed/preprocessed_2025-08-19.jsonl \
  --run-name qual_check \
  --models lead3,textrank,distilbart,bart \
  --limit 100
```

### Fast Evaluation (Skip HF Models)
```bash
python -m model.evaluate \
  --pre-jsonl data/processed/preprocessed_2025-08-19.jsonl \
  --run-name fast_check \
  --models lead3,textrank \
  --fast \
  --limit 100
```

### Evaluation with References
```bash
python -m model.evaluate \
  --pre-jsonl data/processed/preprocessed_2025-08-19.jsonl \
  --ref-field summary \
  --run-name with_refs \
  --models lead3,textrank,distilbart,bart \
  --limit 100
```

### Available Models
- **lead3**: Lead-3 sentences (baseline) - ⚡ Very fast
- **textrank**: TextRank algorithm (custom sentence tokenizer) - ⚡ Fast
- **distilbart**: DistilBART-CNN model - 🐌 Slow (first run downloads model)
- **bart**: BART-Large-CNN model - 🐌 Slow (first run downloads model)

**Note**: HF models (distilbart, bart) are slow on first run due to model downloading. Use `--fast` flag to skip them for quick evaluation.

Results will be saved to `results/<run_name>/` directory.

## 📌 Project Goals
- Scrape tech news from sources like TechCrunch, Wired, and The Verge
- Summarize using pretrained LLMs (Hugging Face Transformers)
- Analyze article trends and summary performance
- Deploy via Streamlit with interactive UI
- Visualize insights using Tableau

## 🗂️ Project Structure
```text
tech-news-summarizer/
├── 📁 config/                    # Configuration files
│   ├── feeds.yml                 # News source configurations
│   └── feeds_finance.yml         # Finance-specific feeds
│
├── 📁 data/                      # Data storage and processing
│   ├── raw/                      # Raw scraped articles
│   │   ├── articles.jsonl
│   │   └── articles_2025-08-16.jsonl
│   └── processed/                # Cleaned and processed data
│       ├── articles_clean.csv
│       ├── articles_clean.jsonl
│       ├── preprocessed_2025-08-19.jsonl
│       └── preprocessed_2025-08-19_manifest.json
│
├── 📁 model/                     # AI/ML model implementations
│   ├── __init__.py
│   ├── evaluate.py               # Model evaluation framework
│   ├── metrics.py                # Evaluation metrics (ROUGE, BLEU, etc.)
│   ├── rag_summarizer.py         # RAG-based summarization system
│   └── summarizers.py            # Multiple summarization models
│
├── 📁 notebooks/                 # Jupyter notebooks for analysis
│   ├── eda_articles.py          # Article data exploration
│   ├── eda_preprocess.py        # Preprocessing analysis
│   └── eda.ipynb                # Main exploratory analysis
│
├── 📁 scripts/                   # Automation and utility scripts
│   ├── __init__.py
│   ├── bootstrap.py              # Project setup and initialization
│   ├── build_embeddings.py       # Generate article embeddings
│   ├── build_faiss_index.py      # Build FAISS vector search index
│   ├── clean_export_preprocess.py # Data cleaning pipeline
│   ├── eval_rag_vs_vanilla.py    # RAG vs. standard summarization comparison
│   ├── rag_batch.py              # Batch RAG processing
│   ├── run_full_scrape.sh        # Full scraping automation script
│   └── scrape_newspaper.py       # News article scraper
│
├── 📁 src/                       # Core source code modules
│   ├── __init__.py
│   ├── cleaning.py               # Data cleaning and preprocessing
│   ├── embeddings.py             # Text embedding generation
│   ├── faiss_store.py            # FAISS vector database operations
│   └── schema.py                 # Data schemas and validation
│
├── 📁 tests/                     # Comprehensive test suite
│   ├── __init__.py
│   ├── test_clean_export_preprocess.py
│   ├── test_cleaning.py
│   ├── test_embeddings.py
│   ├── test_end_to_end_smoke.py
│   ├── test_faiss_store.py
│   ├── test_inference.py
│   ├── test_metrics.py
│   ├── test_rag.py
│   └── test_schema.py
│
├── 📁 venv/                      # Python virtual environment
├── requirements.txt               # Python dependencies
├── setup.py                      # Package installation configuration
├── LICENSE
└── README.md

```

## 🚧 Project Milestones

| Week | Milestone |
|------|-----------|
| ✅ Week 1 | Scraped and cleaned tech articles |
| ✅ Week 2 | Built and evaluated LLM summarizer |
| ✅ Week 3 | RAG-based improvement with FAISS |
| ✅ Week 4 | RAG-based recommendation |
| ✅ Week 5 | Streamlit app deployment + demo |

