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

### Alternative Installation with setup.py

You can also install using the setup script which will automatically download NLTK data:

```bash
pip install -e .
```

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
llm-tech-news-summarizer/
│
├── data/
│   ├── raw/                  # raw scraped data (unprocessed)
│   │   └── articles.jsonl
│   ├── processed/            # cleaned & standardized data
│   │   ├── articles_clean.csv
│   │   └── news.sqlite
│
├── scripts/
│   ├── scrape_newspaper.py   # scrape news articles
│   ├── io_raw.py             # helper functions to save raw data
│   ├── sqlite_raw.py         # store raw data in SQLite
│   ├── cleaner.py            # clean & filter raw data → processed
│   └── ...                   # future utils, e.g., summarizer helpers
│
├── model/
│   ├── summarizers.py        # summarization models (Lead-3, TextRank, BART, etc.)
│   ├── evaluate.py           # evaluation script for comparing models
│   └── test_inference.py     # quick tests for summarizer
│
├── notebooks/
│   ├── eda.ipynb              # exploratory data analysis
│   └── ...
│
├── app/                      # Streamlit or Gradio front-end
│   └── app.py
│
├── config/
│   └── feeds.yml             # list of sources/domains to scrape
│
├── requirements.txt
├── .gitignore
├── README.md
└── LICENSE

```

## 🚧 Project Milestones

| Week | Milestone |
|------|-----------|
| ✅ Week 1 | Scraped and cleaned tech articles |
| 🔄 Week 2 | Built and evaluated LLM summarizer |
| ⏳ Week 3 | RAG-based improvement with FAISS |
| ⏳ Week 4 | Tableau visualizations of trends |
| ⏳ Week 5 | Streamlit app deployment + demo |

