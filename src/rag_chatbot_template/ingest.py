from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from rag_chatbot_template.documents import chunk_text, iter_documents, read_document
from rag_chatbot_template.store import Chunk, VectorStore

app = typer.Typer(help="Ingest TXT/PDF documents into a local vector store.")


def build_store(documents_dir: Path, *, chunk_size: int = 600, overlap: int = 100) -> VectorStore:
    chunks: list[Chunk] = []
    for path in iter_documents(documents_dir):
        text = read_document(path)
        for index, chunk in enumerate(chunk_text(text, chunk_size=chunk_size, overlap=overlap)):
            chunks.append(
                Chunk(
                    id=f"{path.name}:{index}",
                    source=str(path),
                    text=chunk,
                )
            )
    return VectorStore(
        chunks=chunks,
        metadata={
            "documents_dir": str(documents_dir),
            "chunk_size": chunk_size,
            "overlap": overlap,
        },
    )


@app.command()
def ingest(
    documents_dir: Annotated[Path, typer.Argument(help="Directory containing TXT/PDF files.")],
    store: Annotated[Path, typer.Option(help="Output vector store JSON path.")] = Path(
        "vector_store/store.json"
    ),
) -> None:
    vector_store = build_store(documents_dir)
    vector_store.save(store)
    typer.echo(f"Ingested {len(vector_store.chunks)} chunks")
    typer.echo(f"Saved store to {store}")
