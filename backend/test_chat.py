import asyncio
import pytest
from fastapi.testclient import TestClient

from main import app
from core import grounding, llm
from routers import chat as chat_router, documents as documents_router, retrieval as retrieval_router

client = TestClient(app)


@pytest.fixture
def llm_calls(monkeypatch):
    calls = []

    async def fake_generate(system, user_message, history=None):
        calls.append((system, user_message, history))
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
    r = client.post("/chat", json={"message": "apple?"})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"] == "Apples are in the pie [1]."
    assert body["citations"][0]["source"] == "fruit.txt"
    # The model only ever sees the retrieved passages.
    assert "apple apple pie" in llm_calls[0][1]
    assert "ONLY" in llm_calls[0][0]


def test_chat_refuses_without_calling_model_when_nothing_relevant(llm_calls):
    upload("cars.txt", "car car engine")
    r = client.post("/chat", json={"message": "apple?"})
    body = r.json()
    assert body["answer"] == grounding.NO_ANSWER
    assert body["citations"] == []
    assert llm_calls == []


def test_chat_refuses_on_empty_store(llm_calls):
    r = client.post("/chat", json={"message": "apple?"})
    assert r.json()["answer"] == grounding.NO_ANSWER
    assert r.json()["citations"] == []
    assert llm_calls == []


def test_chat_model_says_unknown_clears_sources(monkeypatch):
    async def idk(system, user_message, history=None):
        return grounding.NO_ANSWER

    monkeypatch.setattr(llm, "generate_answer", idk)
    upload("fruit.txt", "apple")
    body = client.post("/chat", json={"message": "apple?"}).json()
    assert body["answer"] == grounding.NO_ANSWER
    assert body["citations"] == []


def test_chat_503_when_llm_unavailable(monkeypatch):
    async def boom(system, user_message, history=None):
        raise llm.LLMUnavailable("no key")

    monkeypatch.setattr(llm, "generate_answer", boom)
    upload("fruit.txt", "apple")
    r = client.post("/chat", json={"message": "apple?"})
    assert r.status_code == 503
    assert r.json()["detail"] == "no key"


def test_chat_validation():
    assert client.post("/chat", json={"message": ""}).status_code == 422
    assert client.post("/chat", json={"message": "  "}).status_code == 400


@pytest.mark.asyncio
async def test_generate_answer_requires_api_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(llm.LLMUnavailable):
        await llm.generate_answer("sys", "hi")


# ---- conversations ----

def test_chat_creates_and_persists_conversation(llm_calls, fake_backends):
    upload("fruit.txt", "apple apple pie")
    body = client.post("/chat", json={"message": "apple?"}).json()
    cid = body["conversation_id"]
    assert cid
    assert body["citations"] == [{"source": "fruit.txt", "page": 1, "score": pytest.approx(1.0)}]

    convo = asyncio.run(fake_backends["conversations"].find_one({"_id": cid}))
    assert [m["role"] for m in convo["messages"]] == ["user", "assistant"]
    assert convo["messages"][0]["content"] == "apple?"
    assert convo["messages"][1]["content"] == "Apples are in the pie [1]."
    assert convo["messages"][1]["citations"][0]["source"] == "fruit.txt"
    assert convo["created_at"] and convo["updated_at"]


def test_chat_follow_up_reuses_conversation_and_sends_history(llm_calls, fake_backends):
    upload("fruit.txt", "apple apple pie")
    cid = client.post("/chat", json={"message": "apple?"}).json()["conversation_id"]
    second = client.post("/chat", json={"message": "more apple?", "conversation_id": cid}).json()
    assert second["conversation_id"] == cid

    history = llm_calls[1][2]
    assert [m["role"] for m in history] == ["user", "assistant"]
    assert history[0]["content"] == "apple?"
    assert "created_at" not in history[0]  # only role/content go to the model

    convo = asyncio.run(fake_backends["conversations"].find_one({"_id": cid}))
    assert len(convo["messages"]) == 4


def test_chat_unknown_conversation_404(llm_calls):
    r = client.post("/chat", json={"message": "apple?", "conversation_id": "nope"})
    assert r.status_code == 404
    assert llm_calls == []


def test_chat_refusal_turn_is_persisted(llm_calls, fake_backends):
    body = client.post("/chat", json={"message": "apple?"}).json()
    convo = asyncio.run(fake_backends["conversations"].find_one({"_id": body["conversation_id"]}))
    assert convo["messages"][1]["content"] == grounding.NO_ANSWER


def test_chat_llm_failure_persists_nothing(monkeypatch, fake_backends):
    async def boom(system, user_message, history=None):
        raise llm.LLMUnavailable("no key")

    monkeypatch.setattr(llm, "generate_answer", boom)
    upload("fruit.txt", "apple")
    assert client.post("/chat", json={"message": "apple?"}).status_code == 503
    assert asyncio.run(fake_backends["conversations"].count_documents({})) == 0
