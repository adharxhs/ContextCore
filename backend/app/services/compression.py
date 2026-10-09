"""Stable HTTP adapter with a deterministic offline demo fallback."""

from __future__ import annotations

import re
from importlib import import_module
from time import perf_counter

from backend.app.api.schemas import CompressRequest

SUPPORTED_SCORERS = {"bm25", "dense", "hybrid", "cross_encoder"}


class CompressionServiceError(Exception):
    """An engine was found but could not provide a contract-valid result."""

    def __init__(self, code: str, message: str, status_code: int = 503) -> None:
        self.code = code
        self.message = message
        self.status_code = status_code
        super().__init__(message)


def _tokens(text: str) -> int:
    return len(text.split())


def _contains_protected_detail(text: str) -> bool:
    """Match the contract's non-negotiable detail types in the demo fallback."""
    lower = text.lower()
    return bool(re.search(r"\d|`{1,3}|\{.*\}|\[.*\]|\b(no|not|never|without)\b", lower))


def _trace(
    chunk_id: str,
    source_type: str,
    source: str | None,
    index: int,
    text: str,
    selected: bool,
    protected: bool,
    score: float,
    reason: str,
) -> dict:
    return {
        "id": chunk_id,
        "source_type": source_type,
        "source": source,
        "original_index": index,
        "text": text,
        "token_count": _tokens(text),
        "selected": selected,
        "protected": protected,
        "score": round(score, 3),
        "reason": reason,
    }


def _offline_compress(request: CompressRequest) -> dict:
    """Keep the Product demo usable until Engine supplies ml.src.inference."""
    started = perf_counter()
    terms = {word.lower().strip(".,?!:;()[]") for word in request.query.split() if len(word) > 2}
    original_index = 0
    candidates = [
        _trace(
            "system:0",
            "system",
            None,
            original_index,
            request.system_prompt,
            True,
            True,
            1.0,
            "protected system instruction",
        )
    ]
    original_index += 1
    for index, message in enumerate(request.history):
        protected = message.role == "user" and index >= max(0, len(request.history) - 2)
        overlap = len(terms & set(message.content.lower().split()))
        candidates.append(
            _trace(
                f"history:{index}",
                "history",
                f"{message.role}:{index}",
                original_index,
                message.content,
                protected,
                protected,
                overlap / max(len(terms), 1),
                "recent user message is protected" if protected else "offline relevance estimate",
            )
        )
        original_index += 1
    for index, block in enumerate(request.context_blocks):
        overlap = len(terms & set(block.content.lower().split()))
        protected = _contains_protected_detail(block.content)
        candidates.append(
            _trace(
                f"context:{block.id}",
                "context",
                block.source or block.id,
                original_index,
                block.content,
                protected,
                protected,
                overlap / max(len(terms), 1),
                "protected number, ID, code, negation, or structured data"
                if protected
                else "offline relevance estimate",
            )
        )
        original_index += 1
    selected = [item for item in candidates if item["protected"]]
    spent = sum(item["token_count"] for item in selected)
    for item in sorted(
        (x for x in candidates if not x["protected"]), key=lambda x: x["score"], reverse=True
    ):
        if spent + item["token_count"] <= request.token_budget:
            item["selected"] = True
            item["reason"] = "fits token budget with highest offline relevance"
            selected.append(item)
            spent += item["token_count"]
        else:
            item["reason"] = "dropped to meet token budget"
    selected.sort(key=lambda x: x["original_index"])
    dropped = [item for item in candidates if not item["selected"]]
    input_tokens = sum(item["token_count"] for item in candidates)
    return {
        "compressed_text": "\n\n".join(item["text"] for item in selected),
        "input_tokens": input_tokens,
        "output_tokens": spent,
        "saved_tokens": max(0, input_tokens - spent),
        "compression_ms": round((perf_counter() - started) * 1000, 2),
        "budget_exceeded": spent > request.token_budget,
        "tokenizer": "heuristic",
        "selected_chunks": selected,
        "dropped_chunks": dropped,
    }


def compress(request: CompressRequest) -> tuple[dict, str]:
    """Run the public engine interface and return its execution mode.

    The fallback exists only when the engine package is unavailable. A
    present-but-failing engine must remain visible to callers instead of being
    mistaken for a real engine result.
    """
    if request.scorer not in SUPPORTED_SCORERS:
        choices = ", ".join(sorted(SUPPORTED_SCORERS))
        raise CompressionServiceError(
            "unsupported_scorer",
            f"Unsupported scorer '{request.scorer}'. Choose one of: {choices}.",
            status_code=400,
        )
    try:
        engine = import_module("ml.src.inference")
    except ModuleNotFoundError as exc:
        if exc.name != "ml.src.inference":
            raise CompressionServiceError(
                "engine_dependency_unavailable",
                "The compression engine is installed but one of its dependencies is unavailable.",
            ) from exc
        return _offline_compress(request), "fallback"
    except Exception as exc:
        raise CompressionServiceError(
            "engine_import_failed", "The compression engine could not be initialized."
        ) from exc

    try:
        result = engine.compress_context(**request.model_dump())
        payload = result.model_dump() if hasattr(result, "model_dump") else result
        if not isinstance(payload, dict):
            raise TypeError("Engine result is not a mapping.")
        # `token_budget` is engine metadata, not part of the public HTTP response.
        # Remove it at this boundary so the API's strict response schema can reject
        # genuinely unknown engine fields without rejecting the documented engine field.
        payload = {key: value for key, value in payload.items() if key != "token_budget"}
        return payload, "engine"
    except ValueError as exc:
        raise CompressionServiceError("engine_rejected_request", str(exc), status_code=400) from exc
    except (AttributeError, TypeError) as exc:
        raise CompressionServiceError(
            "malformed_engine_response",
            "The compression engine returned an unreadable response.",
            status_code=502,
        ) from exc
    except Exception as exc:
        raise CompressionServiceError(
            "engine_execution_failed",
            "The compression engine failed while processing this request.",
        ) from exc
