from fastapi.testclient import TestClient
from main import app
from core.chunker import chunk_text

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_chunk_text_logic():
    text = "A" * 1500
    chunks = chunk_text(text, size=800, overlap=120)
    assert len(chunks) == 3
    assert len(chunks[0]) == 800
    assert len(chunks[1]) == 800
