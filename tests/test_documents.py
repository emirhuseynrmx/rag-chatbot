from __future__ import annotations

from pathlib import Path

import pytest

from rag_chatbot_template.documents import chunk_text, iter_documents, read_document


def test_iter_documents_finds_txt_files() -> None:
    paths = iter_documents(Path("documents"))

    assert Path("documents/company_policy.txt") in paths


def test_read_document_reads_txt() -> None:
    text = read_document(Path("documents/company_policy.txt"))

    assert "Refunds" in text


def test_chunk_text_uses_overlap() -> None:
    chunks = chunk_text("one two three four five six seven", chunk_size=13, overlap=4)

    assert len(chunks) > 1


def test_read_document_rejects_unsupported_file(tmp_path: Path) -> None:
    path = tmp_path / "data.csv"
    path.write_text("hello", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported document type"):
        read_document(path)
