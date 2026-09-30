import uuid
from datetime import datetime
from typing import List, Optional
from fastapi import APIRouter, File, UploadFile, HTTPException, Query
from core.db import documents, chunks
from core.chunker import chunk_text, extract_pages_from_pdf

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    chunk_size: int = Query(800, ge=100, le=4000),
    chunk_overlap: int = Query(120, ge=0, le=500),
):
    """
    Upload a document (PDF or TXT), chunk the content, and save metadata and chunks in MongoDB.
    """
    if chunk_overlap >= chunk_size:
        raise HTTPException(
            status_code=400,
            detail="chunk_overlap must be less than chunk_size."
        )

    content = await file.read()
    filename = file.filename
    document_id = str(uuid.uuid4())

    extracted_pages = []

    if filename.lower().endswith(".pdf"):
        extracted_pages = extract_pages_from_pdf(content)
    elif filename.lower().endswith((".txt", ".md")):
        text = content.decode("utf-8", errors="ignore")
        extracted_pages = [{"page_number": 1, "text": text}]
    else:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file format. Please upload a .pdf, .txt, or .md file."
        )

    all_chunk_docs = []
    total_chunks = 0

    for page_info in extracted_pages:
        page_number = page_info["page_number"]
        page_text = page_info["text"]

        text_chunks = chunk_text(page_text, size=chunk_size, overlap=chunk_overlap)
        for chunk in text_chunks:
            chunk_doc = {
                "document_id": document_id,
                "text": chunk,
                "source": filename,
                "page": page_number,
                "embedding": [],  # Ready for embedding vector generation
                "created_at": datetime.utcnow().isoformat()
            }
            all_chunk_docs.append(chunk_doc)
            total_chunks += 1

    # Insert document metadata
    doc_metadata = {
        "_id": document_id,
        "filename": filename,
        "content_type": file.content_type,
        "total_pages": len(extracted_pages),
        "total_chunks": total_chunks,
        "uploaded_at": datetime.utcnow().isoformat(),
    }

    await documents.insert_one(doc_metadata)

    if all_chunk_docs:
        await chunks.insert_many(all_chunk_docs)

    return {
        "status": "success",
        "message": f"Successfully processed '{filename}' into {total_chunks} chunks.",
        "document": {
            "document_id": document_id,
            "filename": filename,
            "total_pages": len(extracted_pages),
            "total_chunks": total_chunks,
        }
    }


@router.get("/")
async def list_documents():
    """List all uploaded documents."""
    cursor = documents.find({}, {"_id": 1, "filename": 1, "total_pages": 1, "total_chunks": 1, "uploaded_at": 1})
    docs = await cursor.to_list(length=100)
    for doc in docs:
        doc["document_id"] = doc.pop("_id")
    return docs


@router.get("/{document_id}/chunks")
async def get_document_chunks(document_id: str, limit: int = 50):
    """Retrieve chunks belonging to a specific document."""
    cursor = chunks.find({"document_id": document_id}, {"_id": 0}).limit(limit)
    return await cursor.to_list(length=limit)
