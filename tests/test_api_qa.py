from types import SimpleNamespace
import importlib

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def qa_client(monkeypatch):
    from src.rag.schemas import QuestionResponse, TimelineItem

    class DummyRAGTool:
        def __init__(self, *_args, **_kwargs):
            self.vs = SimpleNamespace(_collection=SimpleNamespace(get=lambda **__: {"metadatas": []}))

        def answer_question(self, req):
            assert req.question == "What changed?"
            return QuestionResponse(
                answer="A concise answer (Reuters, 2024-07-15)",
                timeline=[
                    TimelineItem(
                        date="2024-07-15",
                        summary="Key event",
                        source="Reuters",
                        url="https://example.com",
                        citation="(Reuters, 2024-07-15)",
                    )
                ],
                citations=["(Reuters, 2024-07-15)"],
            )

    monkeypatch.setattr("src.rag.tool.RAGTool", DummyRAGTool)

    import src.api.api_main as api_main

    api_main = importlib.reload(api_main)
    client = TestClient(api_main.app)
    return api_main, client


def test_qa_route_success(qa_client):
    api_main, client = qa_client
    payload = {
        "question": "What changed?",
        "temporal_filter": {"natural_language": "since July", "days_back": 90},
        "tickers": ["AAPL"],
        "sources": ["Reuters"],
        "max_events": 4,
        "context_k": 12,
    }

    resp = client.post("/qa", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["answer"].startswith("A concise answer")
    assert data["timeline"][0]["date"] == "2024-07-15"
    assert data["citations"] == ["(Reuters, 2024-07-15)"]


def test_qa_route_error(monkeypatch, qa_client):
    api_main, client = qa_client

    def boom(_):
        raise RuntimeError("boom")

    monkeypatch.setattr(api_main.rag, "answer_question", boom)

    resp = client.post(
        "/qa",
        json={
            "question": "What changed?",
            "temporal_filter": {"days_back": 30},
            "max_events": 4,
            "context_k": 12,
        },
    )

    assert resp.status_code == 500
    assert "boom" in resp.json()["detail"]
