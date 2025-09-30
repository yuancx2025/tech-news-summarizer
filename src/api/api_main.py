from fastapi import FastAPI, HTTPException
import yaml, os
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

@app.get("/healthz")
def healthz():
    return {"status": "ok", "chroma_dir": cfg.get("chroma_dir")}

@app.post("/summarize", response_model=SummarizeResponse)
def summarize(req: SummarizeRequest):
    try:
        return rag.summarize(req)
    except Exception as e:
        print(f"[ERROR] /summarize failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest):
    try:
        return rag.recommend(req)
    except Exception as e:
        print(f"[ERROR] /recommend failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
