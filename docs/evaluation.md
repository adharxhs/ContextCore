# Evaluation

## Benchmark

- 40 version-controlled scenarios in `data/benchmark/cases.jsonl` (see `data/benchmark/README.md`): planted early facts, redundant boilerplate, long histories, code/IDs/numbers, structured data, negations, QA dependencies, and irrelevant distractors.
- Each case defines the query, a token budget that is intended to force ~50% reduction, and `required_evidence` grounded verbatim in the case's own content.
- Fixture tests assert the schema, taxonomy coverage, grounded evidence, and that the budget is below the whitespace input size (`scripts/test_evaluate.py`, `scripts/check_contract.py`).

## Evidence recall

A case passes when every `required_evidence` item is present in `compressed_text` after
whitespace-normalized, case-folded substring matching. Recall is the fraction retained per case;
the summary reports the mean and the number of full-recall cases per scorer. **This exact-match
baseline proves evidence survival, not semantic answer quality**, and must not be reported as the latter.

## Harness

```
python scripts/evaluate.py --scorers bm25,dense,hybrid,cross_encoder --require-tokenizer
```

- Imports only the engine's public entry point `ml.src.inference.compress_context`; no generative LLM call.
- Verifies and records the **engine** tokenizer (`tiktoken cl100k_base`). `--require-tokenizer` exits non-zero if the engine fell back to its heuristic tokenizer, so a wrong tokenizer label cannot pass unnoticed.
- Fixed-budget metrics per scorer: mean evidence recall, full-recall cases, input-token reduction (micro over all tokens and macro per case), p50/p95 compression latency, budget overruns and overrun rate, and errors.
- Emits `data/processed/eval-results.json` (git-ignored) with metadata (timestamp, engine mode, verified tokenizer, case file, platform) plus per-case traces.
- `--mock` exercises only the pipeline with a deterministic keyword selector. Mock output is labelled `mock (pipeline verification only)` and must not be reported as a result.
- Exit code is non-zero when any case fails; `--min-recall <r>` fails if a scorer's mean recall is below `r`.

## Recorded results (real engine)

Command: `python scripts/evaluate.py --scorers bm25,dense,hybrid,cross_encoder --require-tokenizer`

| Field | Value |
|---|---|
| Date | 2026-10-09 |
| Commit | `35b7d01` (merge of engine + product lanes) |
| Platform | Windows-11, Python 3.14.8 |
| Tokenizer | `tiktoken cl100k_base` (verified active) |
| Engine deps | `rank-bm25 0.2.2`, `fastembed 0.9.0`, `onnxruntime 1.31.0`, `numpy 2.5.3`, `scikit-learn 1.9.1` |
| Cases | 40 |
| Dataset revision | `data/benchmark/cases.jsonl` at the commit above |

| scorer | errors | mean recall | full recall | reduction (micro) | p50 ms | p95 ms | over budget |
|---|---|---|---|---|---|---|---|
| bm25 | 0 | 96.2% | 37/40 | 7.1% | 0.750 | 1.291 | 40/40 |
| dense | 0 | 96.2% | 37/40 | 7.1% | 31.725 | 45.579 | 40/40 |
| hybrid | 0 | 96.2% | 37/40 | 7.1% | 31.785 | 49.671 | 40/40 |
| cross_encoder | 0 | 96.2% | 37/40 | 7.1% | 119.730 | 227.723 | 40/40 |

Latencies are a single run on the machine above and vary with load and with first-call model
warm-up; the reduction, recall, and overrun figures are stable across repeated runs.

Missed-evidence cases (all scorers): `qadep-001`, `qadep-002`, `qadep-005` — in each, one of the two
required QA turns (either the assistant answer or its preceding question) was dropped.

### Interpretation

- **The real engine fails budget acceptance.** Every scorer exceeds the token budget on 40/40 cases,
  by ~81 tokens on average, and reduces input tokens by only ~7% instead of the intended ~50%.
- **Scorer choice has no effect.** All four scorers produce identical outputs: 270/309 kept chunks
  (87%) are classified protected, and the selector includes every protected chunk unconditionally.
  In 40/40 cases the protected content alone exceeds the budget, so relevance ranking never gets to
  drop anything. Root cause: over-broad protection patterns in `ml/src/protect.py` (`NUMBER_PATTERN`
  matches any digit, `ID_PATTERN` matches any 8+ char uppercase token, `STRUCTURED_DATA_PATTERN`
  matches any `key: value` line) combined with chunk-level (not span-level) protection and
  unconditional protected selection in `ml/src/selector.py`.
- Evidence recall is 96.2% only because almost nothing is dropped; it is not evidence that
  compression preserves meaning.
- **Do not report the mock numbers** (`data/processed/eval-mock.json`: 92.5% recall, 54% reduction)
  as engine performance. They describe the harness, not the engine.

Reproduce the defect directly:

```
python scripts/evaluate.py --scorers bm25 --require-tokenizer
```

Expect `over` = 40/40 in the summary and a per-scorer overrun line.

## Defects recorded

| # | Severity | Owner | Defect |
|---|---|---|---|
| E1 | High | Engine | Protected chunks exceed the token budget on 100% of cases; budget is unenforceable because 87% of chunks are protected and all are selected. |
| E2 | High | Engine | Protection patterns are far too broad (any digit/token/`key: value` line/negation), causing E1. |
| E3 | Medium | Engine | `ml/requirements.txt` lists `sentence-transformers` (unused) but omits `fastembed`, `numpy`, `scikit-learn`, and `rapidfuzz`, which the code imports; Docker installs the wrong dependencies. |
| E4 | Medium | Engine | `ml/src/selector.py` drops no protected chunk even when over budget; `output_tokens > token_budget` has no top-level signal in `CompressionResult`. |
| E5 | Low | Engine | `ml/tests/test_inference.py:38` budget assertion is `output_tokens <= 120 or saved_tokens >= 0`, which is effectively always true and hides overruns. |
| P1 | Medium | Product | Offline fallback responses are indistinguishable from real engine results; the dashboard cannot label fallback output (contract field `engine` pending). |
| P2 | Low | Product | Product/Engine files fail the configured `ruff` lint (`E501`, `I001`); CI gates validation-owned paths only until fixed. |

## End-to-end quality check

The optional same-model comparison (full versus compressed context) was **not run**: the engine must
first satisfy the budget and protection acceptance criteria above, and a model plus fixed generation
settings must be documented before any answer-quality claim. No quality-preservation claim is made.
