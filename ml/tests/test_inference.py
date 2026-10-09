import pytest
from ml.src.inference import compress_context
from ml.src.tokenizer import count_tokens
from ml.src.types import ContextBlock, Message


def test_compress_context_contract():
    system_prompt = "You are an expert software engineer assistant."
    history = [
        Message(role="user", content="How do I write a binary search in Python?"),
        Message(
            role="assistant",
            content="Here is binary search: ```python\ndef binary_search(arr, target):\n    low, high = 0, len(arr) - 1\n    while low <= high:\n        mid = (low + high) // 2\n        if arr[mid] == target: return mid\n        elif arr[mid] < target: low = mid + 1\n        else: high = mid - 1\n    return -1\n```",
        ),
        Message(role="user", content="What is the time complexity?"),
    ]
    context_blocks = [
        ContextBlock(
            id="doc1",
            content="Binary search runs in O(log n) time complexity because the search interval is halved at each step.",
            source="algorithms.md",
        ),
        ContextBlock(
            id="doc2",
            content="Unrelated paragraph about baking sourdough bread in a Dutch oven at 450 degrees Fahrenheit.",
            source="cooking.md",
        ),
    ]

    result = compress_context(
        system_prompt=system_prompt,
        history=history,
        context_blocks=context_blocks,
        query="What is the time complexity of binary search?",
        token_budget=120,
        scorer="hybrid",
    )

    assert result.input_tokens > 0
    assert result.output_tokens <= 120
    assert result.saved_tokens == result.input_tokens - result.output_tokens
    assert result.compression_ms >= 0.0
    assert len(result.selected_chunks) > 0
    assert result.token_budget == 120
    assert result.tokenizer in ("cl100k_base", "heuristic")

    # Ensure system prompt or relevant context is preserved
    selected_texts = [c.text for c in result.selected_chunks]
    assert any("expert software engineer" in t for t in selected_texts)
    assert any("O(log n)" in t for t in selected_texts)

    # Ensure traces have required fields
    for trace in result.selected_chunks + result.dropped_chunks:
        assert trace.id
        assert trace.source_type in ("system", "history", "context")
        assert trace.original_index >= 0
        assert trace.token_count > 0
        assert isinstance(trace.selected, bool)
        assert isinstance(trace.reason, str)
        assert len(trace.reason) > 0


def test_preserves_original_order():
    history = [
        Message(role="user", content="Step 1: First instruction"),
        Message(role="assistant", content="Step 2: Second instruction"),
        Message(role="user", content="Step 3: Third instruction"),
    ]
    result = compress_context(
        system_prompt="System instructions",
        history=history,
        context_blocks=[],
        query="instruction",
        token_budget=200,
        scorer="bm25",
    )
    indices = [c.original_index for c in result.selected_chunks]
    assert indices == sorted(indices)


def test_accepts_dict_inputs():
    result = compress_context(
        system_prompt="You are a helper.",
        history=[{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi there"}],
        context_blocks=[{"id": "c1", "content": "Sample context block"}],
        query="Hello",
        token_budget=100,
        scorer="dense",
    )
    assert result.input_tokens > 0
    assert result.output_tokens <= 100
    assert result.saved_tokens == result.input_tokens - result.output_tokens


def test_qa_dependency_retention():
    history = [
        Message(role="user", content="What is the retention period for hot storage?"),
        Message(role="assistant", content="Hot storage retention is 90 days."),
        Message(role="user", content="Thanks for the answer."),
    ]
    result = compress_context(
        system_prompt="Governance assistant.",
        history=history,
        context_blocks=[],
        query="hot storage retention",
        token_budget=50,
        scorer="bm25",
    )
    selected_ids = [c.id for c in result.selected_chunks]
    # If assistant message is selected, user question must be retained
    if "history:1:0" in selected_ids:
        assert "history:0:0" in selected_ids
        trace_0 = next(c for c in result.selected_chunks if c.id == "history:0:0")
        assert "QA dependency" in trace_0.reason or trace_0.selected


def test_small_budget_and_protected_overflow():
    system_prompt = "You are a specialized security agent."
    history = [
        Message(role="user", content="What is the token limit?"),
    ]
    # System prompt + recent user turn have ~15 tokens. Budget is 5.
    result = compress_context(
        system_prompt=system_prompt,
        history=history,
        context_blocks=[],
        query="token limit",
        token_budget=5,
        scorer="bm25",
    )
    # Protected content must not be discarded
    assert "specialized security agent" in result.compressed_text
    assert "What is the token limit?" in result.compressed_text
    assert result.output_tokens > 5
    # Negative savings are not clamped
    assert result.saved_tokens == result.input_tokens - result.output_tokens
    # Reasons explain protection overflow
    for trace in result.selected_chunks:
        assert trace.protected
        assert "retained under protection rule despite exceeding token budget" in trace.reason


def test_empty_and_invalid_inputs():
    # Empty inputs
    result = compress_context(
        system_prompt="",
        history=[],
        context_blocks=[],
        query="anything",
        token_budget=100,
    )
    assert result.compressed_text == ""
    assert result.input_tokens == 0
    assert result.output_tokens == 0
    assert result.saved_tokens == 0
    assert result.token_budget == 100
    assert result.tokenizer in ("cl100k_base", "heuristic")

    # Invalid budget
    with pytest.raises(ValueError, match="token_budget must be a positive integer"):
        compress_context(
            system_prompt="test",
            history=[],
            context_blocks=[],
            query="test",
            token_budget=0,
        )

    with pytest.raises(ValueError, match="token_budget must be a positive integer"):
        compress_context(
            system_prompt="test",
            history=[],
            context_blocks=[],
            query="test",
            token_budget=-10,
        )


def test_all_supported_scorers():
    for scorer in ["bm25", "dense", "hybrid", "cross_encoder"]:
        res = compress_context(
            system_prompt="Assistant prompt",
            history=[Message(role="user", content="Question")],
            context_blocks=[
                ContextBlock(id="1", content="Answer about databases"),
                ContextBlock(id="2", content="Unrelated baking recipe"),
            ],
            query="database configuration",
            token_budget=80,
            scorer=scorer,
        )
        assert res.output_tokens <= 80
        assert res.saved_tokens == res.input_tokens - res.output_tokens
        assert count_tokens(res.compressed_text) == res.output_tokens
        assert res.token_budget == 80
        assert res.tokenizer in ("cl100k_base", "heuristic")
