import pytest
from fastapi.testclient import TestClient

from main import app
from routers import documents as documents_router, retrieval as retrieval_router

client = TestClient(app)


def upload(name: str, text: str):
    r = client.post("/documents/upload", files={"file": (name, text.encode(), "text/plain")})
    assert r.status_code == 200, r.text
    return r.json()["document"]["document_id"]


def test_retrieve_ranks_most_similar_first():
    upload("fruit.txt", "apple apple apple pie")
    upload("cars.txt", "car car engine")
    r = client.post("/retrieve", json={"query": "apple", "top_k": 2})
    assert r.status_code == 200
    results = r.json()["results"]
    assert results[0]["source"] == "fruit.txt"
    assert results[0]["score"] > results[1]["score"]


def test_retrieve_filters_by_document():
    upload("fruit.txt", "apple")
    car_id = upload("cars.txt", "car")
    r = client.post("/retrieve", json={"query": "apple", "document_id": car_id})
    assert [x["source"] for x in r.json()["results"]] == ["cars.txt"]


def test_retrieve_empty_store():
    r = client.post("/retrieve", json={"query": "anything"})
    assert r.status_code == 200
    assert r.json()["results"] == []


def test_retrieve_validation():
    assert client.post("/retrieve", json={"query": ""}).status_code == 422
    assert client.post("/retrieve", json={"query": "x", "top_k": 0}).status_code == 422
    assert client.post("/retrieve", json={"query": "   "}).status_code == 400


def test_upload_rejects_oversized_file(monkeypatch):
    monkeypatch.setattr(documents_router, "MAX_UPLOAD_BYTES", 10)
    r = client.post("/documents/upload", files={"file": ("big.txt", b"x" * 50, "text/plain")})
    assert r.status_code == 413
