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

Budgets are derived at fixture time as `max(floor(0.5 * whitespace-input-tokens), evidence-tokens + 12, 24)`.
This guarantees every case demands at least ~50% token reduction while leaving enough
room for the required evidence. Real engine token counts (tiktoken) can differ from the
whitespace estimate; budgets are far enough below input size to remain forcing.

## Categories (40 cases)

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

## Editing

- Keep IDs stable; add new cases with new IDs.
- Every `required_evidence` item must be a substring of the case's own content.
- Keep `token_budget` below input size (see the fixture test).
- Run `python scripts/evaluate.py --mock` from the repo root after editing to sanity-check.