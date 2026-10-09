import pytest
from ml.src.inference import compress_context
from ml.src.types import ContextBlock, Message


def test_contract_fields_present():
    result = compress_context(
        system_prompt="You are a helpful assistant.",
        history=[Message(role="user", content="What is Python?")],
        context_blocks=[ContextBlock(id="c1", content="Python is a programming language")],
        query="Python",
        token_budget=100,
        scorer="bm25",
    )
    assert hasattr(result, "compressed_text")
    assert hasattr(result, "input_tokens")
    assert hasattr(result, "output_tokens")
    assert hasattr(result, "saved_tokens")
    assert hasattr(result, "compression_ms")
    assert hasattr(result, "budget_exceeded")
    assert hasattr(result, "token_budget")
    assert hasattr(result, "tokenizer")
    assert hasattr(result, "selected_chunks")
    assert hasattr(result, "dropped_chunks")


def test_token_budget_echoed():
    budgets = [50, 100, 200, 500]
    for budget in budgets:
        result = compress_context(
            system_prompt="Test",
            history=[Message(role="user", content="Question")],
            context_blocks=[],
            query="test",
            token_budget=budget,
            scorer="bm25",
        )
        assert result.token_budget == budget


def test_tokenizer_field_valid():
    result = compress_context(
        system_prompt="Test",
        history=[Message(role="user", content="Question")],
        context_blocks=[],
        query="test",
        token_budget=100,
        scorer="bm25",
    )
    assert result.tokenizer in ("cl100k_base", "heuristic")


def test_all_chunks_have_reasons():
    result = compress_context(
        system_prompt="You are an assistant.",
        history=[
            Message(role="user", content="Question 1"),
            Message(role="assistant", content="Answer 1"),
            Message(role="user", content="Question 2"),
        ],
        context_blocks=[
            ContextBlock(id="c1", content="Relevant context about the topic"),
            ContextBlock(id="c2", content="Completely unrelated content about cooking"),
        ],
        query="Question 2",
        token_budget=80,
        scorer="hybrid",
    )
    for trace in result.selected_chunks:
        assert trace.reason, f"Selected chunk {trace.id} has no reason"
        assert len(trace.reason.strip()) > 0, f"Selected chunk {trace.id} has empty reason"
    
    for trace in result.dropped_chunks:
        assert trace.reason, f"Dropped chunk {trace.id} has no reason"
        assert len(trace.reason.strip()) > 0, f"Dropped chunk {trace.id} has empty reason"


def test_oversized_protected_chunk_retained():
    very_long_system = "System instructions: " + " critical" * 100
    result = compress_context(
        system_prompt=very_long_system,
        history=[],
        context_blocks=[],
        query="test",
        token_budget=10,
        scorer="bm25",
    )
    assert result.budget_exceeded is True
    assert result.output_tokens > result.token_budget
    assert len(result.selected_chunks) > 0
    assert all(c.protected for c in result.selected_chunks)
    assert any("exceeding token budget" in c.reason for c in result.selected_chunks)


def test_qa_dependency_preservation():
    history = [
        Message(role="user", content="What is the capital of France?"),
        Message(role="assistant", content="The capital of France is Paris."),
        Message(role="user", content="What about Spain?"),
        Message(role="assistant", content="The capital of Spain is Madrid."),
    ]
    result = compress_context(
        system_prompt="Geography assistant",
        history=history,
        context_blocks=[],
        query="capital of Spain",
        token_budget=100,
        scorer="hybrid",
    )
    selected_ids = [c.id for c in result.selected_chunks]
    if "history:3:0" in selected_ids:
        assert "history:2:0" in selected_ids, "Assistant answer selected but user question dropped"
