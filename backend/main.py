from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routers.documents import router as documents_router

app = FastAPI(title="Knowledge Chat API")

app.add_middleware(
    CORSMiddleware,
    # 3000 = Create React App, 5173 = Vite
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(documents_router)


@app.get("/health")
async def health():
    return {"status": "ok"}
