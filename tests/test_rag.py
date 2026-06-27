from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

import rag_chatbot_template.chat as chat_module
import rag_chatbot_template.ingest as ingest_module
from rag_chatbot_template.chat import ask_question
from rag_chatbot_template.ingest import build_store
from rag_chatbot_template.retrieve import retrieve
from rag_chatbot_template.store import VectorStore


def test_build_store_creates_chunks() -> None:
    store = build_store(Path("documents"), chunk_size=120, overlap=20)

    assert store.chunks
    assert store.chunks[0].source.endswith("company_policy.txt")
    assert store.metadata["chunk_size"] == 120


def test_store_save_and_load(tmp_path: Path) -> None:
    store = build_store(Path("documents"), chunk_size=120, overlap=20)
    path = store.save(tmp_path / "store.json")

    loaded = VectorStore.load(path)

    assert len(loaded.chunks) == len(store.chunks)


def test_retrieve_returns_relevant_source() -> None:
    store = build_store(Path("documents"), chunk_size=120, overlap=20)

    results = retrieve("refund order number", store)

    assert results
    assert "Refunds" in results[0].chunk.text


def test_ask_question_returns_source_aware_context() -> None:
    store = build_store(Path("documents"), chunk_size=120, overlap=20)

    answer = ask_question("What does the refund policy say?", store)

    assert "Context" in answer.answer
    assert answer.sources
    assert answer.retrieved_chunks[0]["score"] >= 0


def test_ask_question_handles_empty_store() -> None:
    answer = ask_question("anything", VectorStore(chunks=[]))

    assert "No relevant document chunks" in answer.answer
    assert answer.sources == []
    assert answer.retrieved_chunks == []


def test_ingest_cli_writes_store(tmp_path: Path) -> None:
    store_path = tmp_path / "store.json"

    result = CliRunner().invoke(
        ingest_module.app,
        ["documents", "--store", str(store_path)],
    )

    assert result.exit_code == 0
    assert store_path.exists()


def test_chat_cli_prints_sources(tmp_path: Path) -> None:
    store_path = build_store(Path("documents"), chunk_size=120, overlap=20).save(
        tmp_path / "store.json"
    )

    result = CliRunner().invoke(
        chat_module.app,
        ["refund policy", "--store", str(store_path)],
    )

    assert result.exit_code == 0
    assert "Sources:" in result.output
