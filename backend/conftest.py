import pytest
from mongomock_motor import AsyncMongoMockClient

from routers import (
    chat as chat_router,
    conversations as conversations_router,
    documents as documents_router,
    retrieval as retrieval_router,
)

# Tiny deterministic "embedding": one axis per keyword.
KEYWORDS = ["apple", "car", "moon"]


def fake_vec(text: str):
    t = text.lower()
    return [float(t.count(k)) for k in KEYWORDS]


@pytest.fixture(autouse=True)
def fake_backends(monkeypatch):
    """Replace every MongoDB collection and the embedding model so tests never touch Atlas."""
    mock_db = AsyncMongoMockClient()["test"]
    monkeypatch.setattr(documents_router, "documents", mock_db["documents"])
    monkeypatch.setattr(documents_router, "files", mock_db["files"])
    monkeypatch.setattr(documents_router, "chunks", mock_db["chunks"])
    monkeypatch.setattr(retrieval_router, "chunks", mock_db["chunks"])
    monkeypatch.setattr(chat_router, "conversations", mock_db["conversations"])
    monkeypatch.setattr(conversations_router, "conversations", mock_db["conversations"])
    monkeypatch.setattr(documents_router, "embed_texts", lambda texts: [fake_vec(t) for t in texts])
    monkeypatch.setattr(retrieval_router, "embed_query", fake_vec)
    return mock_db
