from __future__ import annotations

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


def build_answer(question: str, retrieved: list[RetrievedChunk]) -> RagAnswer:
    if not retrieved:
        return RagAnswer(
            question=question,
            answer="No relevant document chunks were found for this question.",
            sources=[],
            retrieved_chunks=[],
        )
    context = "\n\n".join(
        f"[{index}] Source: {item.chunk.source}\n{item.chunk.text}"
        for index, item in enumerate(retrieved, start=1)
    )
    return RagAnswer(
        question=question,
        answer=(
            "Use the retrieved context below to answer the question. "
            "Review the sources before sending a final response.\n\n"
            f"Question: {question}\n\nContext:\n{context}"
        ),
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
    typer.echo(answer.answer)
    if answer.sources:
        typer.echo("\nSources:")
        for source in answer.sources:
            typer.echo(f"- {source}")
