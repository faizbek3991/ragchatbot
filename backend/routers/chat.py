import uuid
from datetime import datetime, timezone
from typing import List
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from core import grounding, llm
from core.db import conversations
from routers.retrieval import search_chunks

router = APIRouter(tags=["chat"])

TOP_K = 5
HISTORY_MESSAGES = 6  # prior messages sent to the model for follow-up questions


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    conversation_id: str | None = None


class Citation(BaseModel):
    source: str
    page: int | None
    score: float


class ChatResponse(BaseModel):
    answer: str
    citations: List[Citation]
    conversation_id: str


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """Answer a message using only retrieved document chunks, and save the turn to the conversation."""
    message = req.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="message must not be blank.")

    # Continue an existing conversation, or start a new one.
    history: List[dict] = []
    if req.conversation_id:
        convo = await conversations.find_one({"_id": req.conversation_id})
        if convo is None:
            raise HTTPException(status_code=404, detail="Conversation not found.")
        history = [
            {"role": m["role"], "content": m["content"]}
            for m in convo["messages"][-HISTORY_MESSAGES:]
        ]
    conversation_id = req.conversation_id or str(uuid.uuid4())

    # retrieve
    found = await search_chunks(message, TOP_K)
    grounded = grounding.build_context(grounding.filter_relevant(found))

    if not grounded["grounded"]:
        # Nothing relevant retrieved: refuse without calling the model at all.
        answer, citations = grounding.NO_ANSWER, []
    else:
        # prompt + generate
        try:
            answer = await llm.generate_answer(
                grounding.SYSTEM_PROMPT,
                grounding.build_user_message(message, grounded["context"]),
                history,
            )
        except llm.LLMUnavailable as e:
            raise HTTPException(status_code=503, detail=str(e))
        known = answer != grounding.NO_ANSWER
        citations = (
            [Citation(source=s["source"], page=s["page"], score=s["score"]) for s in grounded["sources"]]
            if known
            else []
        )

    # persist
    now = _now()
    await conversations.update_one(
        {"_id": conversation_id},
        {
            "$push": {
                "messages": {
                    "$each": [
                        {"role": "user", "content": message, "created_at": now},
                        {
                            "role": "assistant",
                            "content": answer,
                            "citations": [c.model_dump() for c in citations],
                            "created_at": now,
                        },
                    ]
                }
            },
            "$set": {"updated_at": now},
            "$setOnInsert": {"created_at": now},
        },
        upsert=True,
    )

    return ChatResponse(answer=answer, citations=citations, conversation_id=conversation_id)
