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
  "token_budget": 1000,
  "tokenizer": "cl100k_base",
  "selected_chunks": [ <ChunkTrace>, ... ],
  "dropped_chunks": [ <ChunkTrace>, ... ]
}
```

`saved_tokens` equals `input_tokens - output_tokens` (never negative). `budget_exceeded` is `true` when
`output_tokens > token_budget`. `selected_chunks` and `dropped_chunks` are lists of `ChunkTrace` objects (below), not
opaque ids. `compressed_text` is the selected chunks joined in original input order.

### Reported metadata

These fields make the result self-describing. `budget_exceeded` is implemented by the engine
(`ml/src/types.py`, `ml/src/inference.py`). The remaining fields are approved by Validation & Lead
and are still pending in their owning lane; `scripts/check_contract.py` reports the missing ones as
`PEND` (approved, awaiting implementation) rather than failing, and the evaluation harness derives
them itself.

- `budget_exceeded` (Engine, **implemented**): `true` if and only if `output_tokens > token_budget`.
  This only happens when protected content alone cannot fit; it is never `true` for ordinary
  relevance-driven selection. The earlier working name `over_budget` is retired in favour of the
  engine's shipped `budget_exceeded`.
- `token_budget` (Engine, **pending**): the budget the engine enforced, echoed so that
  `budget_exceeded` is self-describing and a consumer does not have to resend it.
- `tokenizer` (Engine, **pending**): the encoder actually used, exactly one of `cl100k_base`
  (tiktoken) or `heuristic` (fallback). The fallback must never be reported as `cl100k_base`.
- `execution_mode` (**API only**, Product, **pending**): `engine` when the real engine served the
  request, `fallback` when the offline demo fallback did. It mirrors the `X-ContextCore-Execution`
  header and is included in the JSON body so HTTP clients that ignore headers still see it. The API
  response schema must also mirror `budget_exceeded`.

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
cannot be loaded, the engine's heuristic fallback must be surfaced in the `tokenizer` field rather
than silently reported as `cl100k_base`; evaluation records the verified tokenizer and must fail a
run that labels the fallback as `cl100k_base`.

### Budget semantics

`output_tokens` should be `<= token_budget`. Protected chunks may exceed the budget only when the
protected content alone exceeds it (removing it would violate a protection rule); in that case
`budget_exceeded` is `true`, each over-budget protected chunk's `reason` must state the protection,
and the evaluation harness must flag the overrun. Over-budget output is a detectable failure, never a
silent result: a consumer must surface `budget_exceeded` (and must not present an over-budget result
as strictly within budget).

## 3. Protection and dependencies

Never silently remove system instructions, recent user messages, numbers, IDs, dates, code blocks, negations, or structured data. A selected assistant response should retain its directly preceding user question when needed for meaning. The trace records every protection or dependency decision. Protection applies to the smallest relevant span; a chunk must not be protected merely because an unrelated part of it matches a pattern.

## 4. HTTP API

| Method | Route | Request | Response |
|---|---|---|---|
| `GET` | `/health` | none | `{"status":"ok"}` |
| `POST` | `/v1/compress` | library inputs | `CompressionResult` |

The API validates all request fields with explicit Pydantic models. Invalid data returns a structured `422`; unsupported scorer returns `400`. No provider key is needed for `/v1/compress`.

When the engine cannot be imported, the API serves an offline fallback so the dashboard stays
usable. Real and fallback responses must be distinguishable: responses carry the header
`X-ContextCore-Execution: engine` (real engine) or `X-ContextCore-Execution: fallback` (offline
demo) **and** the body field `execution_mode` with the same value, and the dashboard must label
fallback output as such. No provider key is needed for `/v1/compress`.

## 5. Evaluation interface

Each benchmark case declares `id`, `category`, inputs, token budget, and `required_evidence`.
Evaluation compares scorers at the same budget and records evidence recall, token reduction,
p50/p95 compression latency, the tokenizer actually used, the execution mode (real engine vs
offline fallback), model-degradation mode for the dense/cross-encoder scorers (whether the ONNX
model loaded or the scorer silently degraded), budget overruns, and failures. The optional
real-LLM quality check uses the same query and fixed model configuration for original and
compressed contexts. Evaluation must not claim answer-quality preservation from evidence recall
alone.

## 6. Ownership

- Engine owner: library input/output and chunk traces.
- Evaluation owner: case format, metrics, and reported results.
- Product owner: API schema mirrors this document and the dashboard consumes only this result shape.
