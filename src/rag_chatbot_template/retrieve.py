from __future__ import annotations

from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from rag_chatbot_template.store import Chunk, VectorStore


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: Chunk
    score: float


def retrieve(question: str, store: VectorStore, *, top_k: int = 3) -> list[RetrievedChunk]:
    if not store.chunks:
        return []
    documents = [chunk.text for chunk in store.chunks]
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform([*documents, question])
    scores = cosine_similarity(matrix[-1], matrix[:-1]).flatten()
    ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)[:top_k]
    return [
        RetrievedChunk(chunk=store.chunks[index], score=round(float(score), 4))
        for index, score in ranked
    ]
