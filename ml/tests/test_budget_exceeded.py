from ml.src.inference import compress_context
from ml.src.types import Message, ContextBlock


def test_budget_exceeded_flag_when_overflow():
    result = compress_context(
        system_prompt="You are a helpful assistant.",
        history=[Message(role="user", content="What is Python?")],
        context_blocks=[
            ContextBlock(id="c1", content="Python is a programming language"),
        ],
        query="Python",
        token_budget=5,
        scorer="bm25",
    )
    assert result.budget_exceeded is True
    assert result.output_tokens > 5


def test_budget_exceeded_flag_when_within_budget():
    result = compress_context(
        system_prompt="Help.",
        history=[Message(role="user", content="Hi")],
        context_blocks=[],
        query="hi",
        token_budget=500,
        scorer="bm25",
    )
    assert result.budget_exceeded is False
    assert result.output_tokens <= 500


def test_budget_exceeded_protected_overflow():
    result = compress_context(
        system_prompt="You are a specialized security agent with critical instructions.",
        history=[Message(role="user", content="What is the token limit for this request?")],
        context_blocks=[],
        query="token limit",
        token_budget=5,
        scorer="bm25",
    )
    assert result.budget_exceeded is True
    assert all(c.protected for c in result.selected_chunks)
    assert any("exceeding token budget" in c.reason for c in result.selected_chunks)


def test_saved_tokens_never_negative():
    result = compress_context(
        system_prompt="System",
        history=[],
        context_blocks=[],
        query="test",
        token_budget=100,
        scorer="bm25",
    )
    assert result.saved_tokens >= 0
    assert result.saved_tokens == max(0, result.input_tokens - result.output_tokens)


def test_budget_exceeded_with_large_context():
    result = compress_context(
        system_prompt="You are an assistant.",
        history=[
            Message(role="user", content="Question 1"),
            Message(role="assistant", content="Answer 1" * 50),
            Message(role="user", content="Question 2"),
        ],
        context_blocks=[
            ContextBlock(id="c1", content="Context block " * 100),
        ],
        query="Question",
        token_budget=50,
        scorer="hybrid",
    )
    assert isinstance(result.budget_exceeded, bool)
    assert result.output_tokens <= 50 or result.budget_exceeded is True
