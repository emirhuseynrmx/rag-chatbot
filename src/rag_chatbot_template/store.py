from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class Chunk(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    source: str
    text: str


class VectorStore(BaseModel):
    model_config = ConfigDict(frozen=True)

    chunks: list[Chunk]
    metadata: dict[str, str | int] = Field(default_factory=dict)

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path) -> VectorStore:
        return cls.model_validate_json(path.read_text(encoding="utf-8"))


def save_json(data: dict[str, object], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path
