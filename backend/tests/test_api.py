from fastapi.testclient import TestClient

from backend.app.main import app

client = TestClient(app)


def payload() -> dict:
    return {"system_prompt": "Never invent account balances.", "history": [],
            "context_blocks": [{"id": "billing", "content": "Refunds take five business days."}],
            "query": "How long does a refund take?", "token_budget": 100, "scorer": "hybrid"}


def test_health() -> None:
    assert client.get("/health").json() == {"status": "ok"}


def test_compress_returns_trace() -> None:
    response = client.post("/v1/compress", json=payload())
    assert response.status_code == 200
    assert response.json()["selected_chunks"][0]["protected"] is True


def test_invalid_budget_is_422() -> None:
    assert client.post("/v1/compress", json=payload() | {"token_budget": 0}).status_code == 422


def test_unknown_scorer_is_400() -> None:
    assert client.post("/v1/compress", json=payload() | {"scorer": "magic"}).status_code == 400
