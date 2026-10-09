import re
from typing import NamedTuple

try:
    from rapidfuzz import fuzz
except ImportError:
    fuzz = None

from ml.src.chunker import InternalChunk


def _normalize_text(text: str) -> str:
    # Lowercase and collapse non-alphanumeric whitespace
    cleaned = re.sub(r"[^\w\s]", "", text.lower())
    return re.sub(r"\s+", " ", cleaned).strip()


def _jaccard_similarity(text1: str, text2: str) -> float:
    words1 = set(_normalize_text(text1).split())
    words2 = set(_normalize_text(text2).split())
    if not words1 or not words2:
        return 0.0
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    return len(intersection) / len(union)


class DedupResult(NamedTuple):
    chunks: list[InternalChunk]
    duplicate_indices: set[int]


def deduplicate_chunks(
    chunks: list[InternalChunk],
    near_dup_threshold: float = 0.88,
) -> DedupResult:
    seen_exact: dict[str, int] = {}  # normalized text -> chunk index
    seen_unique: list[int] = []  # indices of retained chunks for near-dup checks
    dup_indices: set[int] = set()

    for idx, chunk in enumerate(chunks):
        norm = _normalize_text(chunk.trace.text)
        if not norm:
            continue

        # 1. Exact Deduplication
        if norm in seen_exact:
            prev_idx = seen_exact[norm]
            prev_id = chunks[prev_idx].trace.id
            dup_indices.add(idx)
            reason_suffix = f"exact duplicate of chunk '{prev_id}'"
            if chunk.trace.reason:
                chunk.trace.reason = f"{chunk.trace.reason}; {reason_suffix}"
            else:
                chunk.trace.reason = reason_suffix
            continue

        # 2. Near Deduplication
        is_near_dup = False
        for prev_idx in seen_unique:
            prev_chunk = chunks[prev_idx]
            sim = 0.0
            if fuzz is not None:
                sim = fuzz.ratio(norm, _normalize_text(prev_chunk.trace.text)) / 100.0
            else:
                sim = _jaccard_similarity(_normalize_text(chunk.trace.text), _normalize_text(prev_chunk.trace.text))

            if sim >= near_dup_threshold:
                is_near_dup = True
                dup_indices.add(idx)
                reason_suffix = f"near-duplicate of chunk '{prev_chunk.trace.id}' (sim {sim:.2f})"
                if chunk.trace.reason:
                    chunk.trace.reason = f"{chunk.trace.reason}; {reason_suffix}"
                else:
                    chunk.trace.reason = reason_suffix
                break

        if not is_near_dup:
            seen_exact[norm] = idx
            seen_unique.append(idx)

    return DedupResult(chunks=chunks, duplicate_indices=dup_indices)
