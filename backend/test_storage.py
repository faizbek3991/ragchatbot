import asyncio
import pytest
from fastapi.testclient import TestClient

from main import app
from core import llm

client = TestClient(app)


def upload(name: str, content: bytes, content_type: str = "text/plain") -> str:
    r = client.post("/documents/upload", files={"file": (name, content, content_type)})
    assert r.status_code == 200, r.text
    return r.json()["document"]["document_id"]


@pytest.fixture
def fake_llm(monkeypatch):
    async def answer(system, user_message, history=None):
        return "Apples [1]."

    monkeypatch.setattr(llm, "generate_answer", answer)


# ---- original files ----

def test_upload_stores_original_file_and_download_returns_same_bytes(fake_backends):
    original = "apple pie recipe".encode()
    doc_id = upload("fruit.txt", original)

    stored = asyncio.run(fake_backends["files"].find_one({"_id": doc_id}))
    assert bytes(stored["data"]) == original and stored["size"] == len(original)

    r = client.get(f"/documents/{doc_id}/file")
    assert r.status_code == 200
    assert r.content == original
    assert r.headers["content-disposition"].startswith("attachment;")
    assert r.headers["x-content-type-options"] == "nosniff"


def test_download_ignores_client_supplied_content_type():
    doc_id = upload("notes.txt", b"<script>alert(1)</script>", content_type="text/html")
    r = client.get(f"/documents/{doc_id}/file")
    assert r.headers["content-type"].startswith("text/plain")
    assert "attachment" in r.headers["content-disposition"]


def test_download_filename_is_url_encoded():
    doc_id = upload("my file ü.txt", b"x")
    disposition = client.get(f"/documents/{doc_id}/file").headers["content-disposition"]
    assert "my%20file%20%C3%BC.txt" in disposition


def test_download_unknown_document_404():
    assert client.get("/documents/nope/file").status_code == 404


def test_document_list_reports_has_file():
    upload("fruit.txt", b"apple")
    assert client.get("/documents/").json()[0]["has_file"] is True


def test_rejected_upload_stores_nothing(fake_backends):
    r = client.post("/documents/upload", files={"file": ("pic.png", b"x", "image/png")})
    assert r.status_code == 400
    assert asyncio.run(fake_backends["files"].count_documents({})) == 0


# ---- conversations API ----

def test_list_and_get_conversations(fake_llm):
    upload("fruit.txt", b"apple apple pie")
    first = client.post("/chat", json={"message": "apple?"}).json()["conversation_id"]
    second = client.post("/chat", json={"message": "another apple question"}).json()["conversation_id"]

    listing = client.get("/conversations/").json()
    assert [c["conversation_id"] for c in listing] == [second, first]  # newest first
    assert listing[1]["title"] == "apple?"
    assert listing[1]["message_count"] == 2

    detail = client.get(f"/conversations/{first}").json()
    assert [m["role"] for m in detail["messages"]] == ["user", "assistant"]
    assert detail["messages"][1]["content"] == "Apples [1]."
    assert detail["messages"][1]["citations"][0]["source"] == "fruit.txt"


def test_get_unknown_conversation_404():
    assert client.get("/conversations/nope").status_code == 404


def test_list_conversations_empty():
    assert client.get("/conversations/").json() == []
