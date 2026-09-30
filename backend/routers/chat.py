from typing import List, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from core import grounding, llm
from routers.retrieval import search_chunks

router = APIRouter(tags=["chat"])


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int = Field(5, ge=1, le=20)
    document_id: Optional[str] = None


class Source(BaseModel):
    id: int
    source: str
    page: int
    document_id: str
    score: float
    text: str


class ChatResponse(BaseModel):
    answer: str
    grounded: bool
    sources: List[Source]


@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """Answer a question using only retrieved document chunks, with numbered citations."""
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="question must not be blank.")

    found = await search_chunks(question, req.top_k, req.document_id)
    grounded = grounding.build_context(grounding.filter_relevant(found))

    # Nothing relevant retrieved: refuse without calling the model at all.
    if not grounded["grounded"]:
        return ChatResponse(answer=grounding.NO_ANSWER, grounded=False, sources=[])

    try:
        answer = await llm.generate_answer(
            grounding.SYSTEM_PROMPT,
            grounding.build_user_message(question, grounded["context"]),
        )
    except llm.LLMUnavailable as e:
        raise HTTPException(status_code=503, detail=str(e))

    return ChatResponse(
        answer=answer,
        grounded=answer != grounding.NO_ANSWER,
        sources=grounded["sources"] if answer != grounding.NO_ANSWER else [],
    )
