# LLM-Powered Tech News Summarizer 🚀📰

This project builds a full-stack pipeline to scrape, clean, summarize, and visualize tech news using large language models like BART and T5.

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

├── scripts/
│   ├── scrape_newspaper.py   # scrape news articles
│   ├── io_raw.py             # helper functions to save raw data
│   ├── sqlite_raw.py         # store raw data in SQLite
│   ├── cleaner.py            # clean & filter raw data → processed
│   └── ...                   # future utils, e.g., summarizer helpers
│
├── model/
│   ├── summarizer.py         # summarization pipeline (vanilla)
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
| 🔄 Week 1 | Scraped and cleaned tech articles |
| ⏳ Week 2 | Built and evaluated LLM summarizer |
| ⏳ Week 3 | RAG-based improvement with FAISS |
| ⏳ Week 4 | Tableau visualizations of trends |
| ⏳ Week 5 | Streamlit app deployment + demo |

