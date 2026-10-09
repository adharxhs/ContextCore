"""Stable HTTP adapter with a deterministic offline demo fallback."""

from __future__ import annotations

from importlib import import_module
import re
from time import perf_counter

from backend.app.api.schemas import CompressRequest

SUPPORTED_SCORERS = {"bm25", "dense", "hybrid", "cross_encoder"}


def _tokens(text: str) -> int:
    return len(text.split())


def _contains_protected_detail(text: str) -> bool:
    """Match the contract's non-negotiable detail types in the demo fallback."""
    lower = text.lower()
    return bool(re.search(r"\d|`{1,3}|\{.*\}|\[.*\]|\b(no|not|never|without)\b", lower))


def _trace(chunk_id: str, source_type: str, source: str | None, index: int, text: str,
           selected: bool, protected: bool, score: float, reason: str) -> dict:
    return {"id": chunk_id, "source_type": source_type, "source": source,
            "original_index": index, "text": text, "token_count": _tokens(text),
            "selected": selected, "protected": protected, "score": round(score, 3), "reason": reason}


def _offline_compress(request: CompressRequest) -> dict:
    """Keep the Product demo usable until Engine supplies ml.src.inference."""
    started = perf_counter()
    terms = {word.lower().strip(".,?!:;()[]") for word in request.query.split() if len(word) > 2}
    candidates = [_trace("system:0", "system", None, 0, request.system_prompt, True, True, 1.0,
                         "protected system instruction")]
    for index, message in enumerate(request.history):
        protected = message.role == "user" and index >= max(0, len(request.history) - 2)
        overlap = len(terms & set(message.content.lower().split()))
        candidates.append(_trace(f"history:{index}", "history", None, index, message.content, protected,
                                 protected, overlap / max(len(terms), 1),
                                 "recent user message is protected" if protected else "offline relevance estimate"))
    for index, block in enumerate(request.context_blocks):
        overlap = len(terms & set(block.content.lower().split()))
        protected = _contains_protected_detail(block.content)
        candidates.append(_trace(f"context:{block.id}", "context", block.source or block.id, index,
                                 block.content, protected, protected, overlap / max(len(terms), 1),
                                 "protected number, ID, code, negation, or structured data"
                                 if protected else "offline relevance estimate"))
    selected = [item for item in candidates if item["protected"]]
    spent = sum(item["token_count"] for item in selected)
    for item in sorted((x for x in candidates if not x["protected"]), key=lambda x: x["score"], reverse=True):
        if spent + item["token_count"] <= request.token_budget:
            item["selected"] = True
            item["reason"] = "fits token budget with highest offline relevance"
            selected.append(item)
            spent += item["token_count"]
        else:
            item["reason"] = "dropped to meet token budget"
    selected.sort(key=lambda x: (x["source_type"] != "system", x["original_index"]))
    dropped = [item for item in candidates if not item["selected"]]
    input_tokens = sum(item["token_count"] for item in candidates)
    return {"compressed_text": "\n\n".join(item["text"] for item in selected), "input_tokens": input_tokens,
            "output_tokens": spent, "saved_tokens": input_tokens - spent,
            "compression_ms": round((perf_counter() - started) * 1000, 2),
            "selected_chunks": selected, "dropped_chunks": dropped}


def compress(request: CompressRequest) -> tuple[dict, str]:
    """Run the public engine interface and return its execution mode.

    The fallback exists only when the engine package is unavailable. A
    present-but-failing engine must remain visible to callers instead of being
    mistaken for a real engine result.
    """
    if request.scorer not in SUPPORTED_SCORERS:
        choices = ", ".join(sorted(SUPPORTED_SCORERS))
        raise ValueError(f"Unsupported scorer '{request.scorer}'. Choose one of: {choices}.")
    try:
        engine = import_module("ml.src.inference")
        result = engine.compress_context(**request.model_dump())
        payload = result.model_dump() if hasattr(result, "model_dump") else result
        return payload, "engine"
    except ModuleNotFoundError as exc:
        if exc.name != "ml.src.inference":
            raise
        return _offline_compress(request), "fallback"
