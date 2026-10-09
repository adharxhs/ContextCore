from typing import NamedTuple

from ml.src.chunker import InternalChunk
from ml.src.tokenizer import count_tokens
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
    if not chunks:
        return SelectionResult(selected_chunks=[], dropped_chunks=[], compressed_text="")

    def _format_text(indices: set[int]) -> str:
        return "\n\n".join(chunks[i].trace.text for i in sorted(indices) if chunks[i].trace.text.strip())

    def _count_serialized_tokens(indices: set[int]) -> int:
        return count_tokens(_format_text(indices))

    selected_indices: set[int] = set()

    # 1. Select protected chunks (excluding duplicates)
    for idx, chunk in enumerate(chunks):
        if idx in duplicate_indices:
            continue
        if chunk.trace.protected:
            selected_indices.add(idx)
            chunk.trace.selected = True

    protected_tokens = _count_serialized_tokens(selected_indices)

    # 2. If protected content alone fits within budget, greedily add candidate chunks
    if protected_tokens < token_budget:
        # Rank remaining candidate chunks by relevance score descending, tie-break by original_index
        candidates: list[tuple[int, float]] = []
        for idx, chunk in enumerate(chunks):
            if idx in duplicate_indices or idx in selected_indices or chunk.trace.score <= 0.0:
                continue
            candidates.append((idx, chunk.trace.score))

        candidates.sort(key=lambda x: (x[1], -x[0]), reverse=True)

        for idx, score in candidates:
            chunk = chunks[idx]
            needed_parents = [
                p_idx
                for p_idx in chunk.qa_parent_indices
                if p_idx not in selected_indices and p_idx not in duplicate_indices
            ]

            trial_indices = selected_indices | {idx} | set(needed_parents)
            trial_tokens = _count_serialized_tokens(trial_indices)

            if trial_tokens <= token_budget:
                selected_indices = trial_indices
                chunk.trace.selected = True
                if not chunk.trace.reason:
                    chunk.trace.reason = f"selected: high query relevance (score: {score:.2f})"
                else:
                    chunk.trace.reason += f"; selected (score: {score:.2f})"

                for p_idx in needed_parents:
                    p_chunk = chunks[p_idx]
                    p_chunk.trace.selected = True
                    p_reason = f"retained as QA dependency for assistant turn '{chunk.trace.id}'"
                    if p_chunk.trace.reason:
                        p_chunk.trace.reason += f"; {p_reason}"
                    else:
                        p_chunk.trace.reason = p_reason

    # 3. Finalize traces and explanations
    final_output_tokens = _count_serialized_tokens(selected_indices)
    is_protected_overflow = (protected_tokens > token_budget)

    selected_traces: list[ChunkTrace] = []
    dropped_traces: list[ChunkTrace] = []

    for idx, chunk in enumerate(chunks):
        trace = chunk.trace
        if idx in selected_indices:
            trace.selected = True
            if is_protected_overflow and trace.protected:
                if "exceeding token budget" not in trace.reason:
                    trace.reason += "; retained under protection rule despite exceeding token budget"
            selected_traces.append(trace)
        else:
            trace.selected = False
            if idx in duplicate_indices:
                # Reason already populated during dedup
                pass
            elif is_protected_overflow:
                if not trace.reason:
                    trace.reason = f"dropped: token budget exhausted by protected content (score: {trace.score:.2f})"
                else:
                    trace.reason += f"; dropped: token budget exhausted by protected content (score: {trace.score:.2f})"
            elif not trace.reason:
                trace.reason = f"dropped: token budget exhausted (score: {trace.score:.2f})"
            else:
                trace.reason += f"; dropped: token budget exhausted (score: {trace.score:.2f})"
            dropped_traces.append(trace)

    # 4. Order preservation: ensure selected and dropped chunks are sorted by original_index
    selected_traces.sort(key=lambda t: t.original_index)
    dropped_traces.sort(key=lambda t: t.original_index)

    compressed_text = _format_text(selected_indices)

    return SelectionResult(
        selected_chunks=selected_traces,
        dropped_chunks=dropped_traces,
        compressed_text=compressed_text,
    )
