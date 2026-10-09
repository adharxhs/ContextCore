# ContextCore Contract

This document is the shared agreement for the compression engine, API, dashboard, and evaluation. Change it before changing a shared interface. The project name is **ContextCore**; its former working name is retired.

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
  "budget_exceeded": false,
  "selected_chunks": [ <ChunkTrace>, ... ],
  "dropped_chunks": [ <ChunkTrace>, ... ]
}
```

`saved_tokens` equals `input_tokens - output_tokens` (never negative). `budget_exceeded` is `true` when
`output_tokens > token_budget`. `selected_chunks` and `dropped_chunks` are lists of `ChunkTrace` objects (below), not
opaque ids. `compressed_text` is the selected chunks joined in original input order.

Every chunk trace contains:

```json
{
  "id": "history:12:0",
  "source_type": "history",
  "source": "user:12",
  "original_index": 12,
  "text": "...",
  "token_count": 43,
  "selected": true,
  "protected": false,
  "score": 0.81,
  "reason": "high query relevance"
}
```

`score` is a relative ranking value, not a probability.

### Tokenizer requirement

`input_tokens` and `output_tokens` must be produced with `tiktoken` `cl100k_base`. If tiktoken
cannot be loaded, the engine's heuristic fallback must be surfaced rather than silently reported
as `cl100k_base`; evaluation records the verified tokenizer.

### Budget semantics

`output_tokens` should be `<= token_budget`. Protected chunks may exceed the budget only when the
protected content alone exceeds it (removing it would violate a protection rule); in that case each
over-budget protected chunk's `reason` must state the protection, and the evaluation harness must
flag the overrun. Over-budget output is a detectable failure, never a silent result.

## 3. Protection and dependencies

Never silently remove system instructions, recent user messages, numbers, IDs, dates, code blocks, negations, or structured data. A selected assistant response should retain its directly preceding user question when needed for meaning. The trace records every protection or dependency decision. Protection applies to the smallest relevant span; a chunk must not be protected merely because an unrelated part of it matches a pattern.

## 4. HTTP API

| Method | Route | Request | Response |
|---|---|---|---|
| `GET` | `/health` | none | `{"status":"ok"}` |
| `POST` | `/v1/compress` | library inputs | `CompressionResult` |

The API validates all request fields with explicit Pydantic models. Invalid data returns a structured `422`; unsupported scorer returns `400`. No provider key is needed for `/v1/compress`.

When the engine cannot be imported, the API may serve an offline fallback so the dashboard stays
usable, but the response must be distinguishable from a real engine result (the dashboard labels
fallback output). **Pending contract change:** add an `engine` field (`"ml"` or `"offline_fallback"`)
to `CompressionResponse`; this needs Product and Engine sign-off before implementation.

## 5. Evaluation interface

Each benchmark case declares `id`, inputs, token budget, and `required_evidence`. Evaluation compares scorers at the same budget and records evidence recall, token reduction, p50/p95 compression latency, budget overruns, and failures. The optional real-LLM quality check uses the same query and fixed model configuration for original and compressed contexts.

## 6. Ownership

- Engine owner: library input/output and chunk traces.
- Evaluation owner: case format, metrics, and reported results.
- Product owner: API schema mirrors this document and the dashboard consumes only this result shape.
