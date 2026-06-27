from __future__ import annotations

from fastapi.testclient import TestClient

import rag_chatbot_template.app as app_module
from rag_chatbot_template.ingest import build_store


def test_health_endpoint() -> None:
    client = TestClient(app_module.app)

    assert client.get("/health").json() == {"status": "ok"}


def test_ask_endpoint(tmp_path, monkeypatch) -> None:
    store_path = build_store(app_module.Path("documents"), chunk_size=120, overlap=20).save(
        tmp_path / "store.json"
    )
    monkeypatch.setattr(app_module, "DEFAULT_STORE", store_path)
    client = TestClient(app_module.app)

    response = client.post("/ask", json={"question": "refund policy"})

    assert response.status_code == 200
    assert response.json()["sources"]
