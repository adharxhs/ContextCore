# Benchmark cases

`cases.jsonl` is the version-controlled benchmark used by `scripts/evaluate.py`. Each
line is one case; line numbers change on edit, so reference cases by `id`.

## Schema

| Field | Type | Meaning |
|---|---|---|
| `id` | string | Unique, stable case identifier |
| `category` | string | Taxonomy bucket, see below |
| `system_prompt` | string | Protected instructions kept unchanged |
| `history` | list[Message] | `{role, content}` chat turns |
| `context_blocks` | list[Block] | `{id, content, source?}` external context |
| `query` | string | Current user question for relevance scoring |
| `token_budget` | int | Positive, must be below input size so compression is forced |
| `required_evidence` | list[string] | Substrings that must survive compression (normalized match) |

`required_evidence` strings are grounded: every one appears verbatim in the case's own
system prompt, history, or context blocks, so recall is well defined. A fixture test
enforces this, and also that `token_budget` forces reduction.

## Budget derivation

Core budgets are derived at fixture time as
`max(floor(0.5 * whitespace-input-tokens), evidence-tokens + 12, 24)`. This guarantees every case
demands at least ~50% token reduction while leaving enough room for the required evidence. Stress
cases set the budget deliberately below a single protected or oversized sentence so the
protected-only-overflow and oversized-sentence paths are exercised; their budgets are still below
the whitespace input size. Real engine token counts (tiktoken) can differ from the whitespace
estimate; budgets are far enough below input size to remain forcing.

## Categories (50 cases: 40 core + 10 stress)

| Category | Count | Intent |
|---|---|---|
| `planted_fact` | 6 | Decisive fact early, long distractors after |
| `redundant` | 4 | Near-duplicate blocks that must be deduplicated |
| `long_history` | 5 | Answer buried in one older turn of a long conversation |
| `code_ids_numbers` | 6 | Code, identifiers, versions, ports, decline codes must survive |
| `structured_data` | 5 | JSON / CSV / YAML / Markdown / TOML blocks |
| `negations` | 4 | Prohibitions that must never be dropped |
| `qa_dependency` | 5 | Assistant answer needs its preceding user question |
| `distractors` | 5 | One relevant block among five mostly irrelevant ones |
| `near_dedup` | 3 | Near-identical source blocks; must merge to save budget/overrun |
| `oversized_sentence` | 2 | One sentence larger than the budget (protected and unprotected) |
| `protected_overflow` | 2 | Protected content alone exceeds the budget; must be flagged |
| `protected_precision` | 3 | Irrelevant protected spans must not crowd out the real evidence |

The first eight are the stable **core** taxonomy (fixture requires ≥4 cases each). The last four are
the **stress** categories (≥2 each) added to cover known failure modes; they are reported separately
by the harness so they do not silently move the core trend line.

## Editing

- Keep IDs stable; add new cases with new IDs.
- Every `required_evidence` item must be a substring of the case's own content.
- Keep `token_budget` below input size (see the fixture test).
- Run `python scripts/evaluate.py --mock` from the repo root after editing to sanity-check.

## Validation status

The 50 cases satisfy the 30–50-case scope and cover the full taxonomy. A real-engine run
(`docs/evaluation.md`, `2026-10-10`, base `145a25c`, dataset sha256 `d53a6ad2a2bfe5bf`) reaches ~59%
token reduction, 80–83% mean evidence recall (core 80–84%, stress 80%), and 16/50 budget overruns
(all protected-linked, signaled by the engine's `budget_exceeded` field). The stress cases expose
two open Engine gaps: unprotected oversized sentences are dropped (`os-001`) and irrelevant
protected spans crowd out relevant evidence (`pp-001`).
