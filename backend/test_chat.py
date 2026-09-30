import pytest
from fastapi.testclient import TestClient
from mongomock_motor import AsyncMongoMockClient

from main import app
from core import grounding, llm
from routers import chat as chat_router, documents as documents_router, retrieval as retrieval_router

client = TestClient(app)

KEYWORDS = ["apple", "car", "moon"]


def fake_vec(text: str):
    t = text.lower()
    return [float(t.count(k)) for k in KEYWORDS]


@pytest.fixture(autouse=True)
def fake_backends(monkeypatch):
    mock_db = AsyncMongoMockClient()["test"]
    for module in (documents_router, retrieval_router):
        monkeypatch.setattr(module, "chunks", mock_db["chunks"])
    monkeypatch.setattr(documents_router, "documents", mock_db["documents"])
    monkeypatch.setattr(documents_router, "embed_texts", lambda texts: [fake_vec(t) for t in texts])
    monkeypatch.setattr(retrieval_router, "embed_query", fake_vec)


@pytest.fixture
def llm_calls(monkeypatch):
    calls = []

    async def fake_generate(system, user_message):
        calls.append((system, user_message))
        return "Apples are in the pie [1]."

    monkeypatch.setattr(llm, "generate_answer", fake_generate)
    return calls


def upload(name: str, text: str):
    r = client.post("/documents/upload", files={"file": (name, text.encode(), "text/plain")})
    assert r.status_code == 200, r.text


# ---- grounding helpers ----

def test_build_context_empty():
    assert grounding.build_context([]) == {"context": "", "sources": [], "grounded": False}


def test_build_context_numbers_and_cites():
    chunk = {"text": "hello", "source": "a.txt", "page": 2, "document_id": "d1", "score": 0.9}
    out = grounding.build_context([chunk, chunk])
    assert out["grounded"] is True
    assert "[1] (a.txt, page 2)\nhello" in out["context"]
    assert "[2] (a.txt, page 2)" in out["context"]
    assert [s["id"] for s in out["sources"]] == [1, 2]


def test_filter_relevant_drops_low_scores():
    chunks = [{"score": 0.9}, {"score": 0.1}]
    assert grounding.filter_relevant(chunks, min_score=0.5) == [{"score": 0.9}]


# ---- /chat ----

def test_chat_answers_with_sources(llm_calls):
    upload("fruit.txt", "apple apple pie")
    r = client.post("/chat", json={"question": "apple?"})
    assert r.status_code == 200
    body = r.json()
    assert body["grounded"] is True
    assert body["answer"] == "Apples are in the pie [1]."
    assert body["sources"][0]["source"] == "fruit.txt"
    # The model only ever sees the retrieved passages.
    assert "apple apple pie" in llm_calls[0][1]
    assert "ONLY" in llm_calls[0][0]


def test_chat_refuses_without_calling_model_when_nothing_relevant(llm_calls):
    upload("cars.txt", "car car engine")
    r = client.post("/chat", json={"question": "apple?"})
    body = r.json()
    assert body["grounded"] is False
    assert body["answer"] == grounding.NO_ANSWER
    assert body["sources"] == []
    assert llm_calls == []


def test_chat_refuses_on_empty_store(llm_calls):
    r = client.post("/chat", json={"question": "apple?"})
    assert r.json()["grounded"] is False
    assert llm_calls == []


def test_chat_model_says_unknown_clears_sources(monkeypatch):
    async def idk(system, user_message):
        return grounding.NO_ANSWER

    monkeypatch.setattr(llm, "generate_answer", idk)
    upload("fruit.txt", "apple")
    body = client.post("/chat", json={"question": "apple?"}).json()
    assert body["grounded"] is False
    assert body["sources"] == []


def test_chat_503_when_llm_unavailable(monkeypatch):
    async def boom(system, user_message):
        raise llm.LLMUnavailable("no key")

    monkeypatch.setattr(llm, "generate_answer", boom)
    upload("fruit.txt", "apple")
    r = client.post("/chat", json={"question": "apple?"})
    assert r.status_code == 503
    assert r.json()["detail"] == "no key"


def test_chat_validation():
    assert client.post("/chat", json={"question": ""}).status_code == 422
    assert client.post("/chat", json={"question": "  "}).status_code == 400


@pytest.mark.asyncio
async def test_generate_answer_requires_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(llm.LLMUnavailable):
        await llm.generate_answer("sys", "hi")
