from ml.src.chunker import InternalChunk
from ml.src.dedup import deduplicate_chunks
from ml.src.types import ChunkTrace


def _make_chunk(cid: str, text: str, idx: int) -> InternalChunk:
    return InternalChunk(
        trace=ChunkTrace(
            id=cid,
            source_type="context",
            original_index=idx,
            text=text,
            token_count=len(text.split()),
            selected=False,
            protected=False,
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
