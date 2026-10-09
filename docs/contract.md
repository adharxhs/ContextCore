# Context Surgeon Contract

This document is the shared agreement for the compression engine, API, dashboard, and evaluation. Change it before changing a shared interface.

## 1. Library

```python
compress_context(
    system_prompt: str,
    history: list[Message],
    context_blocks: list[ContextBlock],
    query: str,
    token_budget: int,
    scorer: str = "hybrid",
) -> CompressionResult
```

### Input rules

- `system_prompt`: highest-priority instructions; retained unchanged by default.
- `history`: ordered chat messages with `role` (`system`, `user`, or `assistant`) and `content`.
- `context_blocks`: ordered external context with `id`, `content`, and optional `source`.
- `query`: the current user question used for relevance scoring.
- `token_budget`: positive maximum number of output-context tokens.
- `scorer`: one of `bm25`, `dense`, `hybrid`, or `cross_encoder`.

The engine must preserve input order in the final compressed context. It must not make an LLM API call.

## 2. Output

```json
{
  "compressed_text": "...",
  "input_tokens": 4200,
  "output_tokens": 950,
  "saved_tokens": 3250,
  "compression_ms": 84.2,
  "selected_chunks": [],
  "dropped_chunks": []
}
```

`saved_tokens` equals `input_tokens - output_tokens`. Token counts use the configured tokenizer and must state that tokenizer in evaluation results.

Every chunk trace contains:

```json
{
  "id": "history:12",
  "source_type": "history",
  "source": "optional-file-or-block-id",
  "original_index": 12,
  "text": "...",
  "token_count": 43,
  "selected": true,
  "protected": false,
  "score": 0.81,
  "reason": "high query relevance"
}
```

`score` is a relative ranking value, not a probability. Protected chunks may exceed the budget only when removing them would violate a protection rule; the trace must say why.

## 3. Protection and dependencies

Never silently remove system instructions, recent user messages, numbers, IDs, dates, code blocks, negations, or structured data. A selected assistant response should retain its directly preceding user question when needed for meaning. The trace records every protection or dependency decision.

## 4. HTTP API

| Method | Route | Request | Response |
|---|---|---|---|
| `GET` | `/health` | none | `{"status":"ok"}` |
| `POST` | `/v1/compress` | library inputs | `CompressionResult` |

The API validates all request fields with explicit Pydantic models. Invalid data returns a structured `422`; unsupported scorer returns `400`. No provider key is needed for `/v1/compress`.

## 5. Evaluation interface

Each benchmark case declares `id`, inputs, token budget, and `required_evidence`. Evaluation compares scorers at the same budget and records evidence recall, token reduction, p50/p95 compression latency, and failures. The optional real-LLM quality check uses the same query and fixed model configuration for original and compressed contexts.

## 6. Ownership

- Engine owner: library input/output and chunk traces.
- Evaluation owner: case format, metrics, and reported results.
- Product owner: API schema mirrors this document and the dashboard consumes only this result shape.
