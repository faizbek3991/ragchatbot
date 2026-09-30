from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routers.documents import router as documents_router
from routers.retrieval import router as retrieval_router
from routers.chat import router as chat_router
from routers.conversations import router as conversations_router

app = FastAPI(title="Knowledge Chat API")

app.add_middleware(
    CORSMiddleware,
    # 3000 = Create React App, 5173 = Vite
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(documents_router)
app.include_router(retrieval_router)
app.include_router(chat_router)
app.include_router(conversations_router)


@app.get("/health")
async def health():
    return {"status": "ok"}
