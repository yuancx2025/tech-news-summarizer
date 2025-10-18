from datetime import date
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
import yaml, os, json
from fastapi.middleware.cors import CORSMiddleware

from src.rag.tool import RAGTool
from src.rag.schemas import SummarizeRequest, SummarizeResponse, RecommendRequest, RecommendResponse
from dotenv import load_dotenv
load_dotenv()

CFG_PATH = os.getenv("RAG_CONFIG_PATH", "config/rag.yml")
cfg = yaml.safe_load(open(CFG_PATH, "r", encoding="utf-8"))

app = FastAPI(title="News RAG Service")

# Allow local Streamlit
app.add_middleware(
    CORSMiddleware,
    allow_origins=[os.getenv("FRONTEND_ORIGIN", "http://localhost:8501")],
    allow_methods=["*"],
    allow_headers=["*"],
)

rag = RAGTool(cfg)

ANALYTICS_DIR = Path(os.getenv("ANALYTICS_DIR", "data/analytics"))


def _load_metrics(filename: str) -> dict:
    path = ANALYTICS_DIR / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Analytics payload '{filename}' not found")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=500, detail=f"Failed to parse analytics file: {exc}") from exc

@app.get("/healthz")
def healthz():
    return {"status": "ok", "chroma_dir": cfg.get("chroma_dir")}

@app.get("/health")
def health():
    """Alias for /healthz for UI compatibility."""
    return {"status": "ok", "chroma_dir": cfg.get("chroma_dir")}

@app.get("/meta/catalog")
def get_catalog():
    """Return available tickers, sources, sections from Chroma metadata."""
    try:
        # Query all unique metadata values from Chroma
        collection = rag.vs._collection
        
        # Get all metadata (limited sample for performance)
        results = collection.get(limit=5000, include=["metadatas"])
        metadatas = results.get("metadatas", [])
        
        tickers_set = set()
        sources_set = set()
        sections_set = set()
        
        for m in metadatas:
            if m:
                # Tickers
                tickers = m.get("tickers", [])
                if isinstance(tickers, list):
                    tickers_set.update(t.upper() for t in tickers if t)
                # Sources
                source = m.get("source_short")
                if source:
                    sources_set.add(source)
                # Sections
                section = m.get("section")
                if section:
                    sections_set.add(section)
        
        return {
            "tickers": sorted(tickers_set),
            "sources": sorted(sources_set),
            "sections": sorted(sections_set),
        }
    except Exception as e:
        print(f"[WARN] /meta/catalog failed: {e}")
        # Return empty catalog on error
        return {"tickers": [], "sources": [], "sections": []}


@app.get("/analytics/overview")
def analytics_overview(
    start: Optional[date] = Query(None, description="Filter metrics from this date (inclusive)"),
    end: Optional[date] = Query(None, description="Filter metrics up to this date (inclusive)"),
    tickers: Optional[str] = Query(None, description="Comma separated tickers to filter mentions"),
):
    payload = _load_metrics("overview.json")
    sector_rows = payload.get("sector_sentiment", [])
    mention_rows = payload.get("ticker_mentions", [])

    def _in_range(row: dict) -> bool:
        date_str = row.get("date")
        if not date_str:
            return True
        try:
            row_date = date.fromisoformat(str(date_str))
        except ValueError:
            return True
        if start and row_date < start:
            return False
        if end and row_date > end:
            return False
        return True

    filtered_sector = [row for row in sector_rows if _in_range(row)]

    ticker_set = None
    if tickers:
        ticker_set = {t.strip().upper() for t in tickers.split(",") if t.strip()}
        mention_rows = [row for row in mention_rows if row.get("ticker") in ticker_set]

    return {
        "generated_at": payload.get("generated_at"),
        "sector_sentiment": filtered_sector,
        "ticker_mentions": mention_rows,
        "filters": {
            "start": start.isoformat() if start else None,
            "end": end.isoformat() if end else None,
            "tickers": sorted(ticker_set) if ticker_set else None,
        },
    }


@app.get("/analytics/cooccurrence")
def analytics_cooccurrence(
    tickers: Optional[str] = Query(None, description="Comma separated tickers to filter pairs"),
    limit: int = Query(200, ge=1, le=5000, description="Maximum number of pairs to return"),
):
    payload = _load_metrics("ticker_cooccurrence.json")
    rows = payload if isinstance(payload, list) else payload.get("pairs", [])
    ticker_set = None
    if tickers:
        ticker_set = {t.strip().upper() for t in tickers.split(",") if t.strip()}
        rows = [
            row
            for row in rows
            if row.get("source") in ticker_set or row.get("target") in ticker_set
        ]
    return {
        "pairs": rows[:limit],
        "filters": {"tickers": sorted(ticker_set) if ticker_set else None, "limit": limit},
    }

@app.post("/summarize", response_model=SummarizeResponse)
def summarize(req: SummarizeRequest):
    try:
        print(f"[INFO] /summarize request: mode={req.mode}, query={req.query}, days_back={req.days_back}")
        result = rag.summarize(req)
        print(f"[INFO] /summarize success: {len(result.bullets)} bullets, {len(result.sources)} sources")
        return result
    except Exception as e:
        print(f"[ERROR] /summarize failed: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest):
    try:
        return rag.recommend(req)
    except Exception as e:
        print(f"[ERROR] /recommend failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))