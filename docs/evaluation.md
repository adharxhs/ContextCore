# Evaluation

## Benchmark

- 50 version-controlled scenarios in `data/benchmark/cases.jsonl` (see `data/benchmark/README.md`):
  40 **core** cases across the original eight taxonomy buckets, plus 10 **stress** cases added to
  cover known failure modes (`near_dedup`, `oversized_sentence`, `protected_overflow`,
  `protected_precision`).
- Core cases cover planted early facts, redundant boilerplate, long histories, code/IDs/numbers,
  structured data, negations, QA dependencies, and irrelevant distractors. Stress cases target the
  specific failures recorded below.
- Each case defines the query, a token budget that forces compression, and `required_evidence`
  grounded verbatim in the case's own content.
- Fixture tests assert the schema, taxonomy coverage, grounded evidence, and that the budget is
  below the whitespace input size (`scripts/test_evaluate.py`, `scripts/check_contract.py`).

## Evidence recall

A case passes when every `required_evidence` item is present in `compressed_text` after
whitespace-normalized, case-folded substring matching. Recall is the fraction retained per case.
The summary reports the mean, the number of full-recall cases, and a core/stress split plus a
per-category breakdown. **This exact-match baseline proves evidence survival, not semantic answer
quality**, and must not be reported as the latter.

## Harness

```
python scripts/evaluate.py --scorers bm25,dense,hybrid,cross_encoder --require-tokenizer
```

- Imports only the engine's public entry point `ml.src.inference.compress_context`; no generative LLM call.
- Verifies and records the **engine** tokenizer (`tiktoken cl100k_base`). `--require-tokenizer`
  exits non-zero if the engine fell back to its heuristic tokenizer.
- Records **model mode** for `dense`/`hybrid`/`cross_encoder`: whether the FastEmbed ONNX model
  loaded (`full`) or the scorer silently degraded to TF-IDF / hybrid ranking (`degraded`). A
  degraded run must not be reported as a model-backed result.
- Fixed-budget metrics per scorer: mean evidence recall (overall, core, stress, per category),
  full-recall cases, input-token reduction (micro over all tokens and macro per case), p50/p95
  compression latency, budget overruns with a protected-linked flag, and errors.
- Reads the engine's explicit `budget_exceeded` field (contract v2, shipped by Engine in PR #8),
  falling back to computing `output_tokens > token_budget` if absent.
- Classifies every over-budget case as `over_budget` plus `over_budget_protected` (whether the
  retained protected tokens explain it).
- Emits `data/processed/eval-results.json` (git-ignored) with metadata (timestamp, engine mode,
  verified tokenizer, model modes, case file, platform) plus per-case traces.
- Exit-code flags: `--require-tokenizer`, `--fail-on-degraded`, `--fail-on-over-budget`,
  `--min-recall <r>`; any exception also returns non-zero. `--mock` exercises only the pipeline
  with a deterministic keyword selector and must not be reported as a result.

## Recorded results (real engine, working tree after Engine PR #8 merge)

Command: `python scripts/evaluate.py --scorers bm25,dense,hybrid,cross_encoder --require-tokenizer --fail-on-degraded`

| Field | Value |
|---|---|
| Date | 2026-10-10 |
| Base commit | `145a25c` (includes Engine audit PR #8) |
| Platform | Windows-11, Python 3.14.8 |
| Tokenizer | `tiktoken cl100k_base` (verified active) |
| Engine deps | `rank-bm25 0.2.2`, `fastembed 0.9.0`, `onnxruntime 1.31.0`, `numpy 2.5.3`, `scikit-learn 1.9.1`, `rapidfuzz 3.14.6`, `tiktoken 0.14.0` |
| Model mode | dense/hybrid/cross_encoder: **full** (ONNX models loaded); bm25: lexical |
| Cases | 50 (40 core + 10 stress) |
| Dataset revision | `data/benchmark/cases.jsonl` sha256 `d53a6ad2a2bfe5bf` |
| Harness revision | `scripts/evaluate.py` sha256 `0abfd4f4ca324f5d` |

| scorer | errors | mean recall | recall core | recall stress | full recall | reduction (micro) | p50 ms | p95 ms | over budget |
|---|---|---|---|---|---|---|---|---|---|
| bm25 | 0 | 83.0% | 83.8% | 80.0% | 41/50 | 59.3% | 0.700 | 1.421 | 16/50 |
| cross_encoder | 0 | 81.0% | 81.2% | 80.0% | 40/50 | 59.6% | 115.455 | 172.824 | 16/50 |
| dense | 0 | 82.0% | 82.5% | 80.0% | 41/50 | 59.1% | 32.605 | 52.841 | 16/50 |
| hybrid | 0 | 80.0% | 80.0% | 80.0% | 40/50 | 58.8% | 32.570 | 51.218 | 16/50 |

Latencies are a single run on the machine above and vary with load and first-call model warm-up.
All 16 over-budget cases per scorer are protected-linked (`over_budget_protected`). The engine's
explicit `budget_exceeded` field is now present in every result and matches the computed
over-budget detection.

### Stress-category results (identical across scorers)

| category | cases | mean recall | behavior |
|---|---|---|---|
| `near_dedup` | 3 | 100% | near-duplicate blocks merged (2 dropped per case); evidence survives |
| `oversized_sentence` | 2 | 50% | protected oversized sentence retained + flagged over budget; unprotected oversized sentence dropped (recall lost) |
| `protected_overflow` | 2 | 100% | protected-only overflow retained, over budget flagged, evidence survives |
| `protected_precision` | 3 | 66.7% | one case loses the relevant unprotected evidence to an irrelevant protected block |

Missed-evidence cases:

- `bm25`: `hist-002`, `hist-003`, `hist-005`, `qadep-001`, `qadep-003`, `qadep-004`, `qadep-005`, `os-001`, `pp-001` (9 cases).
- `dense`: the same plus `qadep-002`, minus `hist-002` (9 cases).
- `hybrid`: `hist-002`, `hist-003`, `hist-005`, `qadep-001`, `qadep-002`, `qadep-003`, `qadep-004`, `qadep-005`, `os-001`, `pp-001` (10 cases).
- `cross_encoder`: `hist-001`, `hist-002`, `hist-003`, `hist-005`, `qadep-001`, `qadep-003`, `qadep-004`, `qadep-005`, `os-001`, `pp-001` (10 cases).

Budget-overrun cases (identical for all four scorers, 16): `fact-001`, `fact-002`, `dup-002`,
`dup-003`, `dup-004`, `hist-003`, `hist-005`, `code-003`, `code-004`, `struct-001`, `os-002`,
`po-001`, `po-002`, `pp-001`, `pp-002`, `pp-003`. The first ten are the pre-existing core
overruns (mean +4.4 tokens); the six stress overruns are protected-only overflows or precision
over-retention and dominate the +6.5-token overall mean.

### Interpretation

- **Recalibrated benchmark.** The core 40 cases reproduce the earlier trend: ~80-84% evidence
  recall, ~59% token reduction, and the same 10 core budget overruns. The stress cases lower the
  headline reduction (to ~59%) because protected-overflow cases intentionally emit over-budget
  output; always read core vs stress separately.
- **Near-dedup is healthy** when `rapidfuzz` is installed: all near-duplicate stress cases and
  `dup-001`/`dup-003` merge as intended. Without `rapidfuzz` the Jaccard fallback does not merge at
  the 0.88 threshold, which is why the recorded results require `rapidfuzz` (CI installs it).
- **Oversized sentences remain a real gap.** A required sentence larger than the budget that is not
  covered by a protection pattern is silently dropped (`os-001` recall 0 for every scorer). A
  protected oversized sentence is retained but exceeds the budget (`os-002`).
- **Protected-span precision is still coarse.** An irrelevant block that merely contains a
  date/number/negation is force-retained and can crowd out the relevant unprotected evidence
  (`pp-001` recall 0) or push the output over budget (`pp-001`/`pp-002`/`pp-003`).
- **Evidence is genuinely lost on QA dependencies and long history** (`qadep-*`, `hist-*`), where
  the assistant answer or its preceding question is dropped. Exact-evidence recall remains a
  retention baseline, not a semantic-quality claim.
- **Model mode is full** on this machine (ONNX models cached); the harness now fails with
  `--fail-on-degraded` if that is not true, so degraded numbers cannot be presented as model-backed.
- **Contract v2 `budget_exceeded` is now surfaced by the engine** for all 16 over-budget cases
  (engine PR #8). The harness reads this field and reports it as the source.
- **Do not report the mock numbers** (`data/processed/eval-mock.json`) as engine performance.

Reproduce (offline, BM25 only — no model download):

```
python scripts/evaluate.py --scorers bm25 --require-tokenizer --fail-on-degraded
```

Full four-scorer run (requires the FastEmbed models, cached as documented in `data/README.md`):

```
python scripts/evaluate.py --scorers bm25,dense,hybrid,cross_encoder --require-tokenizer --fail-on-degraded
```

## Defects recorded

| # | Severity | Owner | Defect | Status |
|---|---|---|---|---|
| E1 | High → Med | Engine | Protected chunks exceeded the budget on 40/40 cases (pre-fix); residual 10/40 core overruns (+4.4 tokens mean) | **Fixed** by `956f322`; residual open |
| E2 | High | Engine | Protection patterns far too broad (any digit / 8-char uppercase token / `key: value` line) | **Fixed** by `956f322` |
| E3 | High | Engine | `ml/requirements.txt` lists unused `sentence-transformers` and omits `fastembed`/`numpy`/`scikit-learn`/`rapidfuzz`; the backend image therefore installs the wrong deps and the container's dense/hybrid path degrades | **Fixed** by Engine PR #8 (`d92b77b`): manifest now pins correct deps |
| E4 | Medium | Engine | No top-level over-budget signal in `CompressionResult` | **Fixed** by Engine PR #8: `budget_exceeded: bool` field shipped |
| E5 | Low | Engine | `ml/tests/test_inference.py` budget assertion was effectively always true | **Fixed** (now `output_tokens <= budget`) |
| E6 | Medium | Engine | Evidence loss on `qa_dependency`/`long_history` cases (required turn dropped); reproduced on the stress cases too | Open |
| E7 | High → Closed | Engine | `ml/tests/test_dedup.py::test_near_dedup_threshold` | **Closed**: passes with `rapidfuzz` installed (CI installs it); only the Jaccard fallback path still misses the 0.88 threshold |
| E8 | Medium | Engine | Unprotected sentence larger than the budget cannot be retained or truncated; silently dropped (`os-001`) | New, open |
| E9 | Medium | Engine | Protected-span precision: irrelevant protected context is force-retained, crowding out relevant unprotected evidence (`pp-001`) and inflating output (`pp-001..003`) | New, open |
| P1 | Medium | Product | Offline fallback indistinguishable from real engine results | **Fixed** (`X-ContextCore-Execution` header + dashboard labels fallback); contract v2 additionally requests the body field `execution_mode` |
| P2 | Low | Product/Engine | Repository-wide Ruff failures | **Fixed**: Ruff now passes across the repository |
| P3 | Low | Product | API response parity for `execution_mode` and `budget_exceeded` | **Fixed**: the response body and execution header are covered by API tests |

## End-to-end quality check

The optional same-model comparison (full versus compressed context) was **not run**: the engine must
first satisfy the budget and protection acceptance criteria above, and a model plus fixed generation
settings must be documented before any answer-quality claim. No quality-preservation claim is made.

Cross-lane blockers and hand-off status are tracked in `docs/integration-status.md`.
