from fastapi import FastAPI, HTTPException
import yaml, os
from fastapi.middleware.cors import CORSMiddleware

from src.rag.tool import RAGTool
from src.rag.schemas import (
    SummarizeRequest,
    SummarizeResponse,
    RecommendRequest,
    RecommendResponse,
    QuestionRequest,
    QuestionResponse,
)
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

@app.post("/summarize", response_model=SummarizeResponse)
def summarize(req: SummarizeRequest):
    try:
        print(f"[INFO] /summarize request: mode={req.mode}, query={req.query}, days_back={req.days_back}")
        result = rag.summarize(req)
        print(f"[INFO] /summarize success: {len(result.bullets)} bullets")
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


@app.post("/qa", response_model=QuestionResponse)
def qa(req: QuestionRequest):
    try:
        print(
            f"[INFO] /qa question='{req.question}' tickers={req.tickers} sources={req.sources}"
        )
        result = rag.answer_question(req)
        return result
    except Exception as e:
        print(f"[ERROR] /qa failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))