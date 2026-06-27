from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel, Field

from rag_chatbot_template.chat import RagAnswer, ask_question
from rag_chatbot_template.store import VectorStore

app = FastAPI(title="RAG Chatbot Template")
DEFAULT_STORE = Path("vector_store/store.json")


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    top_k: int = Field(default=3, ge=1, le=10)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/ask")
def ask(request: AskRequest) -> RagAnswer:
    store = VectorStore.load(DEFAULT_STORE)
    return ask_question(request.question, store, top_k=request.top_k)
