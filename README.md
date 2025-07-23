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
llm‑tech‑news‑summarizer/
├── data/
│   ├── raw/               # Unprocessed scraped articles (JSON or HTML)
│   └── processed/         # Cleaned data in CSV/SQLite
├── model/
│   ├── summarizer.py      # Summarization pipeline wrapper
│   └── test_inference.py  # Manual testing script
├── scripts/
│   ├── scraper.py         # Scrape from RSS or newspaper3k
│   └── cleaner.py         # Clean and standardize scraped articles
├── notebooks/
│   └── eda.ipynb          # Exploratory analysis notebook
├── app/                   # Streamlit or Gradio UI (Week 5)
├── .gitignore
├── requirements.txt
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

