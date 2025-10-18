from types import SimpleNamespace

import pytest
from langchain.schema import Document

from src.rag.schemas import QuestionRequest
from src.rag.tool import RAGTool


@pytest.fixture
def rag_cfg():
    return {
        "chroma_dir": "tests/chroma",
        "embedding_model": "dummy-emb",
        "llm_model": "dummy-llm",
        "retrieval": {
            "k_final": 8,
            "fetch_k": 20,
            "lambda_mult": 0.6,
            "days_back_start": 30,
            "days_back_max": 365,
            "language": "en",
        },
        "summarize": {},
        "recommend": {},
    }


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")


@pytest.fixture
def rag_tool(monkeypatch, rag_cfg):
    monkeypatch.setattr("src.rag.tool.get_embeddings", lambda **_: object())

    class DummyChroma:
        def __init__(self, *_, **__):
            self._collection = SimpleNamespace(get=lambda **__: {"ids": [], "documents": [], "metadatas": []})

        def as_retriever(self, *_, **__):
            class DummyRetriever:
                def invoke(self, *_args, **__):
                    return []

            return DummyRetriever()

    class DummyLLM:
        def invoke(self, *_args, **__):
            return SimpleNamespace(content="{}")

    monkeypatch.setattr("src.rag.tool.Chroma", DummyChroma)
    monkeypatch.setattr("src.rag.tool.make_llm", lambda **__: DummyLLM())

    tool = RAGTool(rag_cfg)
    return tool


def test_answer_question_sorts_timeline_and_citations(monkeypatch, rag_tool):
    captured = {}

    def fake_retrieve(query, cfg, **kwargs):
        captured["kwargs"] = kwargs
        return [
            Document(page_content="doc1", metadata={"source_short": "Reuters"}),
            Document(page_content="doc2", metadata={"source_short": "WSJ"}),
        ]

    monkeypatch.setattr("src.rag.tool.retrieve_with_config", fake_retrieve)
    monkeypatch.setattr(
        "src.rag.tool.resolve_temporal_filter",
        lambda *_args, **__kwargs: ("2024-05-01", "2024-07-31", 120),
    )

    def fake_run_qa_chain(_map, _reduce, question, docs, max_events):
        assert question.startswith("What")
        assert len(docs) == 2
        return (
            "Answer text (Reuters, 2024-06-01)",
            [
                {
                    "date": "2024-07-15",
                    "summary": "Later event",
                    "source": "Reuters",
                    "url": "https://reuters.com/a",
                    "citation": "(Reuters, 2024-07-15)",
                },
                {
                    "date": "2024-06-01",
                    "summary": "Earlier event",
                    "source": "WSJ",
                    "url": "https://wsj.com/b",
                    "citation": "(WSJ, 2024-06-01)",
                },
            ],
            ["(Reuters, 2024-07-15)", "(WSJ, 2024-06-01)"]
        )

    monkeypatch.setattr("src.rag.tool.run_qa_chain", fake_run_qa_chain)

    req = QuestionRequest(question="What happened?", max_events=3, context_k=12)
    resp = rag_tool.answer_question(req)

    assert resp.answer.startswith("Answer text")
    # Timeline sorted ascending by date
    assert [item.date for item in resp.timeline] == ["2024-06-01", "2024-07-15"]
    # Citations deduplicated and preserved
    assert resp.citations == ["(Reuters, 2024-07-15)", "(WSJ, 2024-06-01)"]

    kwargs = captured["kwargs"]
    assert kwargs["start_date"] == "2024-05-01"
    assert kwargs["end_date"] == "2024-07-31"
    assert kwargs["k_final"] == 12
    assert kwargs["fetch_k"] >= 12
