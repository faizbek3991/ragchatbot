from typing import List
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from core.db import conversations

router = APIRouter(prefix="/conversations", tags=["conversations"])

TITLE_LENGTH = 60


class ConversationSummary(BaseModel):
    conversation_id: str
    title: str
    message_count: int
    updated_at: str


class StoredMessage(BaseModel):
    role: str
    content: str
    citations: List[dict] = []
    created_at: str


class ConversationDetail(BaseModel):
    conversation_id: str
    created_at: str
    updated_at: str
    messages: List[StoredMessage]


@router.get("/", response_model=List[ConversationSummary])
async def list_conversations(limit: int = 50):
    """List saved conversations, most recently active first."""
    limit = max(1, min(limit, 200))
    cursor = conversations.find({}).sort("updated_at", -1).limit(limit)
    return [
        ConversationSummary(
            conversation_id=c["_id"],
            title=c["messages"][0]["content"][:TITLE_LENGTH] if c["messages"] else "(empty)",
            message_count=len(c["messages"]),
            updated_at=c["updated_at"],
        )
        for c in await cursor.to_list(length=limit)
    ]


@router.get("/{conversation_id}", response_model=ConversationDetail)
async def get_conversation(conversation_id: str):
    """Return the full message history of one conversation."""
    convo = await conversations.find_one({"_id": conversation_id})
    if convo is None:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    return ConversationDetail(
        conversation_id=convo["_id"],
        created_at=convo["created_at"],
        updated_at=convo["updated_at"],
        messages=convo["messages"],
    )
