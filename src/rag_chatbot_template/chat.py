from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated

import typer
from pydantic import BaseModel, ConfigDict

from rag_chatbot_template.retrieve import RetrievedChunk, retrieve
from rag_chatbot_template.store import VectorStore

app = typer.Typer(help="Ask questions against a local RAG vector store.")


class RagAnswer(BaseModel):
    model_config = ConfigDict(frozen=True)

    question: str
    answer: str
    sources: list[str]
    retrieved_chunks: list[dict[str, str | float]]
    provider: str = "retrieval-only"


def _call_openai(question: str, context: str) -> str | None:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None
    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful business assistant. "
                        "Answer using ONLY the provided context. "
                        "If the answer is not in the context, say so briefly."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {question}",
                },
            ],
            max_tokens=512,
            temperature=0.2,
        )
        return response.choices[0].message.content
    except Exception:
        return None


def _call_gemini(question: str, context: str) -> str | None:
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        return None
    try:
        import google.generativeai as genai

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        prompt = (
            "You are a helpful business assistant. "
            "Answer using ONLY the provided context. "
            "If the answer is not in the context, say so briefly.\n\n"
            f"Context:\n{context}\n\nQuestion: {question}"
        )
        response = model.generate_content(prompt)
        return response.text
    except Exception:
        return None


def _generate_answer(question: str, context: str) -> tuple[str, str]:
    """Try OpenAI first, then Gemini, then return retrieval-only context."""
    answer = _call_openai(question, context)
    if answer:
        return answer, "openai/gpt-4o-mini"

    answer = _call_gemini(question, context)
    if answer:
        return answer, "gemini/gemini-1.5-flash"

    fallback = (
        "Retrieved context "
        "(set OPENAI_API_KEY or GEMINI_API_KEY for AI-generated answers):\n\n"
        + context
    )
    return fallback, "retrieval-only"


def build_answer(question: str, retrieved: list[RetrievedChunk]) -> RagAnswer:
    if not retrieved:
        return RagAnswer(
            question=question,
            answer="No relevant document chunks were found for this question.",
            sources=[],
            retrieved_chunks=[],
            provider="retrieval-only",
        )

    context = "\n\n".join(
        f"[{index}] Source: {item.chunk.source}\n{item.chunk.text}"
        for index, item in enumerate(retrieved, start=1)
    )

    answer, provider = _generate_answer(question, context)

    return RagAnswer(
        question=question,
        answer=answer,
        sources=sorted({item.chunk.source for item in retrieved}),
        retrieved_chunks=[
            {
                "id": item.chunk.id,
                "source": item.chunk.source,
                "score": item.score,
                "preview": item.chunk.text[:240],
            }
            for item in retrieved
        ],
        provider=provider,
    )


def ask_question(question: str, store: VectorStore, *, top_k: int = 3) -> RagAnswer:
    return build_answer(question, retrieve(question, store, top_k=top_k))


@app.command()
def ask(
    question: Annotated[str, typer.Argument(help="Question to ask.")],
    store: Annotated[Path, typer.Option(help="Vector store JSON path.")] = Path(
        "vector_store/store.json"
    ),
    top_k: Annotated[int, typer.Option(help="Number of chunks to retrieve.")] = 3,
) -> None:
    answer = ask_question(question, VectorStore.load(store), top_k=top_k)
    if answer.provider != "retrieval-only":
        typer.echo(f"[{answer.provider}] {answer.answer}")
    else:
        typer.echo(answer.answer)
    if answer.sources:
        typer.echo("\nSources:")
        for source in answer.sources:
            typer.echo(f"- {source}")
