from ml.src.inference import compress_context
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
    assert result.output_tokens <= 120 or result.saved_tokens >= 0
    assert result.saved_tokens == result.input_tokens - result.output_tokens
    assert result.compression_ms >= 0.0
    assert len(result.selected_chunks) > 0

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
