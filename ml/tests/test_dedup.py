from ml.src.chunker import InternalChunk
from ml.src.dedup import deduplicate_chunks
from ml.src.types import ChunkTrace


def _make_chunk(cid: str, text: str, idx: int, protected: bool = False) -> InternalChunk:
    return InternalChunk(
        trace=ChunkTrace(
            id=cid,
            source_type="context",
            original_index=idx,
            text=text,
            token_count=len(text.split()),
            selected=False,
            protected=protected,
            score=0.0,
            reason="",
        ),
        qa_parent_indices=[],
    )


def test_exact_and_near_dedup():
    c1 = _make_chunk("c1", "Context compression removes redundant tokens from history.", 0)
    c2 = _make_chunk("c2", "Context compression removes redundant tokens from history.", 1)
    c3 = _make_chunk("c3", "Context compression removes redundant tokens from history!", 2)
    c4 = _make_chunk("c4", "Something completely different and unique.", 3)

    result = deduplicate_chunks([c1, c2, c3, c4])
    assert 1 in result.duplicate_indices
    assert 2 in result.duplicate_indices
    assert 3 not in result.duplicate_indices
    assert 0 not in result.duplicate_indices

    # Reasons populated for duplicates
    assert "exact duplicate" in result.chunks[1].trace.reason
    assert "duplicate" in result.chunks[2].trace.reason
    assert result.chunks[0].trace.reason == ""
    assert result.chunks[3].trace.reason == ""


def test_near_dedup_threshold():
    c1 = _make_chunk("c1", "Migration requires approval from security team.", 0)
    c2 = _make_chunk("c2", "Migration needs approval from the security team.", 1)
    c3 = _make_chunk("c3", "Data migration will be scheduled for next week.", 2)

    # Near-dup with default threshold 0.88
    result = deduplicate_chunks([c1, c2, c3])
    assert 1 in result.duplicate_indices
    assert 2 not in result.duplicate_indices

    # Near-dup with strict threshold 0.99
    result_strict = deduplicate_chunks([c1, c2, c3], near_dup_threshold=0.99)
    assert 1 not in result_strict.duplicate_indices
    assert 2 not in result_strict.duplicate_indices


def test_dedup_empty_and_single():
    result_empty = deduplicate_chunks([])
    assert result_empty.duplicate_indices == set()
    assert result_empty.chunks == []

    c1 = _make_chunk("c1", "Single chunk", 0)
    result_single = deduplicate_chunks([c1])
    assert result_single.duplicate_indices == set()
    assert len(result_single.chunks) == 1


def test_dedup_preserves_protected_status():
    c1 = _make_chunk("c1", "Protected important chunk.", 0, protected=True)
    c2 = _make_chunk("c2", "Protected important chunk.", 1, protected=True)

    result = deduplicate_chunks([c1, c2])
    assert 1 in result.duplicate_indices
    assert result.chunks[0].trace.protected is True
    assert result.chunks[1].trace.protected is True
