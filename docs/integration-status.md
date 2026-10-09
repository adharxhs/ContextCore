# Integration status

Owner: Validation & Lead. Last updated 2026-10-10 (base commit `145a25c`, includes Engine PR #8
`d92b77b`). This is the single page for cross-lane blockers; defect ids reference
`docs/evaluation.md`.

## Contract v2 (reconciled)

Validation & Lead reconciled the contract with Engine PR #8. Engine shipped `budget_exceeded: bool`
(the working name `over_budget` is retired). Remaining fields are still pending:

| Field | Owner | Purpose | State |
|---|---|---|---|
| `budget_exceeded` | Engine | explicit over-budget indicator (protected-only overflow) | **IMPLEMENTED** (PR #8) |
| `token_budget` | Engine | echo the enforced budget so `budget_exceeded` is self-describing | `PEND` |
| `tokenizer` | Engine | report the encoder actually used (`cl100k_base` vs `heuristic`) | `PEND` |
| `budget_exceeded` | Product | mirror the engine's field in the HTTP response body | `PEND` |
| `execution_mode` | Product | mirror the `X-ContextCore-Execution` header in the JSON body | `PEND` |

`scripts/check_contract.py` reports these as `[PEND]` (not a failure) until implemented, then
validates their presence and type strictly. The evaluation harness already derives all of them
(and reads `budget_exceeded` from the engine), so validation is not blocked.

## Blockers by lane

### Engine (lane 1)

| Defect | Blocker | Required action |
|---|---|---|
| E6 | Evidence lost on `qadep-*` and `hist-*` (required turn dropped) | Strengthen QA-dependency retention and ranking (bidirectional retention now shipped in PR #8, but recall gap persists) |
| E8 | Unprotected single sentence larger than the budget is silently dropped (`os-001`) | Keep and flag it (over budget) or truncate; never silently drop |
| E9 | Irrelevant protected spans are force-retained and crowd out relevant unprotected evidence (`pp-001`, `pp-002`, `pp-003`) | Apply protection at the smallest relevant span so an unrelated date/number/negation does not protect a whole block |

**Fixed in PR #8 (`d92b77b`):** E3 (manifest now pins `fastembed`, `numpy`, `scikit-learn`, `rapidfuzz`, `tiktoken`); E4 (`budget_exceeded` field); E7 (near-dedup passes with `rapidfuzz` installed).

### Product (lane 2)

| Defect | Blocker | Required action |
|---|---|---|
| P3 | `CompressionResponse` has no `execution_mode` body field (header only) and no `budget_exceeded` field | Add both fields per contract v2 |
| P2 | Product files fail `ruff` (part of 29 repo-wide errors) | Lint the Product lane paths |

### Validation & Lead (lane 3) — this lane

- Contract v2 reconciled with Engine PR #8; `budget_exceeded` is the shipped name, remaining fields `[PEND]`.
- Benchmark expanded to 50 cases (40 core + 10 stress) covering near-dedup, oversized sentences,
  protected-only overflow, and protected-span precision.
- Harness reports tokenizer/model mode, core/stress and per-category recall, and protected-linked
  overruns; `--fail-on-degraded` / `--fail-on-over-budget` gates added.
- Docs, acknowledgements, runbook, and Docker smoke script updated.

## Reproducible validation commands and outcomes (2026-10-10, commit `145a25c`)

| Command | Outcome |
|---|---|
| `python -m pytest` | 66 passed |
| `python -m ruff check scripts` | All checks passed |
| `python scripts/check_contract.py` | PASS; 4 approved fields `PEND` |
| `python scripts/evaluate.py --scorers bm25,dense,hybrid,cross_encoder --require-tokenizer --fail-on-degraded` | 0 errors; recall 80-83% (core 80-84%, stress 80%); reduction ~59%; 16/50 over budget, all protected-linked and engine-flagged; model mode full |
| `bash scripts/smoke.sh` | Reproducible command (Docker); not run here — Docker unavailable on the validation host. Gated to manual CI dispatch. |

Full numbers live in `docs/evaluation.md`. Raw results: `data/processed/eval-results.json`
(git-ignored).

## Handoff notes

- Engine and Product can implement the remaining contract v2 fields independently; `check_contract.py`
  will flip the `PEND` entries to strict checks automatically.
- Once P3 is fixed, the Docker smoke job can move to the normal push/PR path, and
  `--fail-on-over-budget` can gate the core case set.
- No answer-quality claim is made until the optional same-model comparison is configured, run, and
  recorded (see `docs/evaluation.md`).