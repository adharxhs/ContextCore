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

## Recorded results (real engine, after engine fix `956f322`)

Command: `python scripts/evaluate.py --scorers bm25,dense,hybrid,cross_encoder --require-tokenizer`

| Field | Value |
|---|---|
| Date | 2026-10-09 |
| Commit | `a074088` (includes engine fix `956f322` merged via PR #6) |
| Platform | Windows-11, Python 3.14.8 |
| Tokenizer | `tiktoken cl100k_base` (verified active) |
| Engine deps | `rank-bm25 0.2.2`, `fastembed 0.9.0`, `onnxruntime 1.31.0`, `numpy 2.5.3`, `scikit-learn 1.9.1` |
| Cases | 40 |
| Dataset revision | `data/benchmark/cases.jsonl` at the commit above |

| scorer | errors | mean recall | full recall | reduction (micro) | p50 ms | p95 ms | over budget |
|---|---|---|---|---|---|---|---|
| bm25 | 0 | 82.5% | 32/40 | 61.8% | 0.995 | 1.729 | 10/40 |
| dense | 0 | 85.0% | 33/40 | 61.3% | 32.275 | 46.467 | 10/40 |
| hybrid | 0 | 82.5% | 32/40 | 61.1% | 33.165 | 52.204 | 10/40 |
| cross_encoder | 0 | 81.2% | 32/40 | 62.1% | 120.885 | 173.932 | 10/40 |

Latencies are a single run on the machine above and vary with load and with first-call model
warm-up; the reduction, recall, and overrun figures were stable across repeated runs.

Missed-evidence cases: heavily concentrated in `qa_dependency` (`qadep-001`, `qadep-003`,
`qadep-004`, `qadep-005`), `long_history` (`hist-002`, `hist-003`, `hist-005`), and `redundant`
(`dup-001`); `dense` additionally retains `hist-002` (7 missed cases), the other scorers miss 8.
In each missed QA case, either the assistant answer or its preceding question is dropped.

### Interpretation

- **Engine fix `956f322` resolved the earlier blocker.** Protection patterns were narrowed, token
  accounting was corrected, and the BM25 scorer was fixed. Token reduction is now ~61% (above the
  ~50% target) and budget overruns dropped from 40/40 to 10/40 cases; the four scorers now produce
  genuinely different rankings.
- **Remaining budget overruns (10/40, identical across scorers):** `fact-001`, `fact-002`, `dup-002`,
  `dup-003`, `dup-004`, `hist-003`, `hist-005`, `code-003`, `code-004`, `struct-001`. Mean overrun is
  +4.4 tokens; the overshoot comes from protected content the selector keeps, so strict budget
  semantics are still not met on these cases (defect E4/E1-residual).
- **Evidence is now genuinely lost.** Mean recall is 81.2-85.0% — the compression drops required
  evidence. Exact-evidence recall remains a retention baseline, not a semantic-quality claim.
- **One Engine-lane test fails on `main`:** `ml/tests/test_dedup.py::test_near_dedup_threshold`
  (near-duplicate sentences are not merged at the 0.88 threshold). CI now includes `ml/tests`, so
  the suite currently fails.
- **Do not report the mock numbers** (`data/processed/eval-mock.json`: 92.5% recall, 54% reduction)
  as engine performance. They describe the harness, not the engine.

Reproduce:

```
python scripts/evaluate.py --scorers bm25,dense,hybrid,cross_encoder --require-tokenizer
```

Expect `over` = 10/40 in the summary and an overrun line per scorer.

## Defects recorded

| # | Severity | Owner | Defect | Status |
|---|---|---|---|---|
| E1 | High → Med | Engine | Protected chunks exceeded the budget on 40/40 cases (pre-fix) | **Fixed** by `956f322`; residual 10/40 overruns (+4.4 tokens mean) remain |
| E2 | High | Engine | Protection patterns far too broad (any digit / 8-char uppercase token / `key: value` line) | **Fixed** by `956f322` |
| E3 | Medium | Engine | `ml/requirements.txt` lists unused `sentence-transformers`, omits `fastembed`/`numpy`/`scikit-learn`/`rapidfuzz`; Docker installs the wrong deps | Open |
| E4 | Medium | Engine | No top-level over-budget signal in `CompressionResult`; protected chunks are never dropped even when over budget | Open |
| E5 | Low | Engine | `ml/tests/test_inference.py` budget assertion was effectively always true | **Fixed** (now `output_tokens <= budget`) |
| E6 | Medium | Engine | Evidence loss on `qa_dependency`/`long_history`/`redundant` cases: ~15-19% mean recall lost; QA-dependency retention and ranking drop the required turn | New, open |
| E7 | High | Engine | `ml/tests/test_dedup.py::test_near_dedup_threshold` fails on `main` (near-duplicates at 0.88 not merged) | New, open |
| P1 | Medium | Product | Offline fallback indistinguishable from real engine results | **Fixed** (`X-ContextCore-Execution` header + dashboard labels fallback) |
| P2 | Low | Product | Product/Engine files fail `ruff` (`E501`, `I001`); CI gates validation-owned paths only | Open |

## End-to-end quality check

The optional same-model comparison (full versus compressed context) was **not run**: the engine must
first satisfy the budget and protection acceptance criteria above, and a model plus fixed generation
settings must be documented before any answer-quality claim. No quality-preservation claim is made.
