from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient

from backend.app.main import app

client = TestClient(app, raise_server_exceptions=False)


def payload() -> dict:
    return {
        "system_prompt": "Never invent account balances.",
        "history": [],
        "context_blocks": [{"id": "billing", "content": "Refunds take five business days."}],
        "query": "How long does a refund take?",
        "token_budget": 100,
        "scorer": "hybrid",
    }


def test_health() -> None:
    assert client.get("/health").json() == {"status": "ok"}


def test_compress_returns_trace() -> None:
    response = client.post("/v1/compress", json=payload())
    assert response.status_code == 200
    assert response.json()["selected_chunks"][0]["protected"] is True
    assert response.headers["X-ContextCore-Execution"] == "engine"
    body = response.json()
    assert set(body) == {
        "compressed_text",
        "input_tokens",
        "output_tokens",
        "saved_tokens",
        "compression_ms",
        "selected_chunks",
        "dropped_chunks",
    }
    assert {
        "id",
        "source_type",
        "original_index",
        "text",
        "token_count",
        "selected",
        "protected",
        "score",
        "reason",
    } <= set(body["selected_chunks"][0])


def test_invalid_budget_is_422() -> None:
    response = client.post("/v1/compress", json=payload() | {"token_budget": 0})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_request"


def test_unknown_scorer_is_400() -> None:
    response = client.post("/v1/compress", json=payload() | {"scorer": "magic"})
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "unsupported_scorer"


def test_engine_failure_is_not_replaced_by_fallback() -> None:
    with patch(
        "backend.app.services.compression.import_module", side_effect=RuntimeError("engine failed")
    ):
        response = client.post("/v1/compress", json=payload())

    assert response.status_code == 503
    assert response.headers["X-ContextCore-Execution"] == "engine"
    assert response.json()["error"]["code"] == "engine_import_failed"


def test_missing_engine_is_explicitly_marked_as_fallback() -> None:
    missing = ModuleNotFoundError("No module named 'ml.src.inference'")
    missing.name = "ml.src.inference"
    with patch("backend.app.services.compression.import_module", side_effect=missing):
        response = client.post("/v1/compress", json=payload())

    assert response.status_code == 200
    assert response.headers["X-ContextCore-Execution"] == "fallback"
    assert response.headers["X-ContextCore-Tokenizer"] == "heuristic"
    assert (
        response.json()["selected_chunks"][-1]["original_index"]
        > response.json()["selected_chunks"][0]["original_index"]
    )


def test_malformed_engine_result_is_a_structured_502() -> None:
    engine = SimpleNamespace(compress_context=lambda **_: {"compressed_text": "missing fields"})
    with patch("backend.app.services.compression.import_module", return_value=engine):
        response = client.post("/v1/compress", json=payload())

    assert response.status_code == 502
    assert response.headers["X-ContextCore-Execution"] == "engine"
    assert response.json()["error"]["code"] == "invalid_engine_response"
