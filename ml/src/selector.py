from typing import NamedTuple

from ml.src.chunker import InternalChunk
from ml.src.types import ChunkTrace


class SelectionResult(NamedTuple):
    selected_chunks: list[ChunkTrace]
    dropped_chunks: list[ChunkTrace]
    compressed_text: str


def select_chunks(
    chunks: list[InternalChunk],
    duplicate_indices: set[int],
    token_budget: int,
) -> SelectionResult:
    selected_indices: set[int] = set()
    total_tokens = 0

    # 1. Select protected chunks (excluding duplicates)
    for idx, chunk in enumerate(chunks):
        if idx in duplicate_indices:
            continue
        if chunk.trace.protected:
            selected_indices.add(idx)
            total_tokens += chunk.trace.token_count
            chunk.trace.selected = True

    # 2. Rank remaining candidate chunks by relevance score descending
    candidates: list[tuple[int, float]] = []
    for idx, chunk in enumerate(chunks):
        if idx in duplicate_indices or idx in selected_indices:
            continue
        candidates.append((idx, chunk.trace.score))

    # Sort descending by score, stable sort preserving original order for ties
    candidates.sort(key=lambda x: x[1], reverse=True)

    # 3. Greedily select top-scoring chunks that fit within the budget
    for idx, score in candidates:
        chunk = chunks[idx]
        needed_tokens = chunk.trace.token_count

        # Check QA dependencies if this chunk has parent user chunks not yet selected
        needed_parents: list[int] = []
        for p_idx in chunk.qa_parent_indices:
            if p_idx not in selected_indices and p_idx not in duplicate_indices:
                needed_parents.append(p_idx)
                needed_tokens += chunks[p_idx].trace.token_count

        if total_tokens + needed_tokens <= token_budget:
            # Select chunk and its needed parents
            selected_indices.add(idx)
            chunk.trace.selected = True
            total_tokens += chunk.trace.token_count
            if not chunk.trace.reason:
                chunk.trace.reason = f"selected: high query relevance (score: {score:.2f})"
            else:
                chunk.trace.reason += f"; selected (score: {score:.2f})"

            for p_idx in needed_parents:
                selected_indices.add(p_idx)
                p_chunk = chunks[p_idx]
                p_chunk.trace.selected = True
                total_tokens += p_chunk.trace.token_count
                p_reason = f"retained as QA dependency for assistant turn '{chunk.trace.id}'"
                if p_chunk.trace.reason:
                    p_chunk.trace.reason += f"; {p_reason}"
                else:
                    p_chunk.trace.reason = p_reason

    # 4. Finalize trace and reasons for unselected chunks
    selected_traces: list[ChunkTrace] = []
    dropped_traces: list[ChunkTrace] = []

    for idx, chunk in enumerate(chunks):
        trace = chunk.trace
        if idx in selected_indices:
            trace.selected = True
            selected_traces.append(trace)
        else:
            trace.selected = False
            if idx in duplicate_indices:
                # Reason already populated during dedup
                pass
            elif trace.protected and total_tokens > token_budget:
                trace.reason += "; dropped due to budget constraint"
            elif not trace.reason:
                trace.reason = f"dropped: token budget exhausted (score: {trace.score:.2f})"
            else:
                trace.reason += f"; dropped: token budget exhausted (score: {trace.score:.2f})"
            dropped_traces.append(trace)

    # 5. Order preservation: ensure selected chunks are sorted by original_index
    selected_traces.sort(key=lambda t: t.original_index)
    dropped_traces.sort(key=lambda t: t.original_index)

    # 6. Format compressed_text
    compressed_text = "\n\n".join(t.text for t in selected_traces)

    return SelectionResult(
        selected_chunks=selected_traces,
        dropped_chunks=dropped_traces,
        compressed_text=compressed_text,
    )
