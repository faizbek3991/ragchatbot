from typing import List, Optional
import numpy as np
from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field
from core.db import chunks
from core.embeddings import embed_query

router = APIRouter(tags=["retrieval"])

MAX_CANDIDATES = 5000


class RetrieveRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(5, ge=1, le=50)
    document_id: Optional[str] = None


class RetrievedChunk(BaseModel):
    text: str
    source: str
    page: int
    document_id: str
    score: float


class RetrieveResponse(BaseModel):
    query: str
    results: List[RetrievedChunk]


async def search_chunks(query: str, top_k: int, document_id: Optional[str] = None) -> List[dict]:
    """Return the top_k chunks most similar to the query, best first, each with a `score`."""
    query_vec = np.asarray(await run_in_threadpool(embed_query, query), dtype=np.float32)

    filt = {"embedding.0": {"$exists": True}}  # skip chunks that have no embedding
    if document_id:
        filt["document_id"] = document_id

    cursor = chunks.find(filt, {"_id": 0}).limit(MAX_CANDIDATES)
    candidates = await cursor.to_list(length=MAX_CANDIDATES)
    if not candidates:
        return []

    matrix = np.asarray([c["embedding"] for c in candidates], dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1) * np.linalg.norm(query_vec)
    scores = matrix @ query_vec / np.where(norms == 0, 1, norms)

    top = np.argsort(scores)[::-1][:top_k]
    return [
        {
            "text": candidates[i]["text"],
            "source": candidates[i]["source"],
            "page": candidates[i]["page"],
            "document_id": candidates[i]["document_id"],
            "score": float(scores[i]),
        }
        for i in top
    ]


@router.post("/retrieve", response_model=RetrieveResponse)
async def retrieve(req: RetrieveRequest):
    """Embed the query and return the top_k most similar chunks (cosine similarity)."""
    query = req.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="query must not be blank.")

    results = await search_chunks(query, req.top_k, req.document_id)
    return RetrieveResponse(query=query, results=[RetrievedChunk(**r) for r in results])
