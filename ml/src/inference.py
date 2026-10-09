import time
from typing import Any

from ml.src.chunker import chunk_inputs
from ml.src.dedup import deduplicate_chunks
from ml.src.scorers.factory import get_scorer
from ml.src.selector import select_chunks
from ml.src.tokenizer import count_tokens
from ml.src.types import (
    CompressionResult,
    ContextBlock,
    Message,
)


def _compute_input_tokens(
    system_prompt: str,
    history: list[Message],
    context_blocks: list[ContextBlock],
) -> int:
    parts: list[str] = []
    if system_prompt and system_prompt.strip():
        parts.append(system_prompt.strip())
    for m in history:
        if m.content and m.content.strip():
            parts.append(m.content.strip())
    for b in context_blocks:
        if b.content and b.content.strip():
            parts.append(b.content.strip())

    full_uncompressed = "\n\n".join(parts)
    return count_tokens(full_uncompressed)


def _normalize_messages(history: list[Message | dict[str, Any]]) -> list[Message]:
    normalized: list[Message] = []
    for item in history:
        if isinstance(item, Message):
            normalized.append(item)
        elif isinstance(item, dict):
            normalized.append(Message(**item))
        else:
            raise ValueError(f"Invalid Message item: {item}")
    return normalized


def _normalize_context_blocks(
    context_blocks: list[ContextBlock | dict[str, Any]],
) -> list[ContextBlock]:
    normalized: list[ContextBlock] = []
    for item in context_blocks:
        if isinstance(item, ContextBlock):
            normalized.append(item)
        elif isinstance(item, dict):
            normalized.append(ContextBlock(**item))
        else:
            raise ValueError(f"Invalid ContextBlock item: {item}")
    return normalized


def compress_context(
    system_prompt: str,
    history: list[Message],
    context_blocks: list[ContextBlock],
    query: str,
    token_budget: int,
    scorer: str = "hybrid",
) -> CompressionResult:
    start_time = time.perf_counter()

    if token_budget <= 0:
        raise ValueError("token_budget must be a positive integer.")

    norm_history = _normalize_messages(history or [])
    norm_context = _normalize_context_blocks(context_blocks or [])

    # Total input tokens
    input_tokens = _compute_input_tokens(
        system_prompt=system_prompt,
        history=norm_history,
        context_blocks=norm_context,
    )

    # 1. Chunking & Protection Analysis
    chunks = chunk_inputs(
        system_prompt=system_prompt,
        history=norm_history,
        context_blocks=norm_context,
    )

    if not chunks:
        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
        return CompressionResult(
            compressed_text="",
            input_tokens=input_tokens,
            output_tokens=0,
            saved_tokens=input_tokens,
            compression_ms=elapsed_ms,
            selected_chunks=[],
            dropped_chunks=[],
        )

    # 2. Exact and Near Deduplication
    dedup_result = deduplicate_chunks(chunks)

    # 3. Relevance Scoring
    scorer_instance = get_scorer(scorer)
    texts_to_score = [c.trace.text for c in dedup_result.chunks]
    scores = scorer_instance.score(query=query, texts=texts_to_score)

    for chunk, s in zip(dedup_result.chunks, scores, strict=True):
        chunk.trace.score = s

    # 4. Token Budget Selection with QA Dependency retention and order preservation
    selection = select_chunks(
        chunks=dedup_result.chunks,
        duplicate_indices=dedup_result.duplicate_indices,
        token_budget=token_budget,
    )

    # 5. Output calculation
    output_tokens = count_tokens(selection.compressed_text)
    saved_tokens = input_tokens - output_tokens
    elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)

    return CompressionResult(
        compressed_text=selection.compressed_text,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        saved_tokens=saved_tokens,
        compression_ms=elapsed_ms,
        selected_chunks=selection.selected_chunks,
        dropped_chunks=selection.dropped_chunks,
    )
