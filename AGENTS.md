# AGENTS.md

## Goal
Build **Context Surgeon**: a model-agnostic Python library and thin HTTP API that takes a system prompt, history, context blocks, and a query, then returns shorter, query-relevant context **without an extra generative LLM call**. A dashboard demonstrates the result; one real chat model is used only for full-context versus compressed-context evaluation.

## Scope
- Keep: protected spans; chunking; exact/near deduplication; BM25 + dense scoring; cross-encoder reranking; token-budget selection; original ordering; simple question-answer dependency retention; provenance trace.
- Protect by default: system instructions, recent user messages, numbers, dates, IDs, code, negations, and structured data.
- Measure: token reduction, compression latency, evidence recall, and answer-quality preservation on 30–50 reproducible cases.
- Defer: generative summarization, custom model training/fusion, Provence, multi-provider integrations, hosted auth/storage, and production-scale deployment.

## Working rules
- Build the smallest reliable, explainable demo. Compression must work offline except for the optional evaluation adapter.
- Do not silently discard protected content. Every selected or dropped chunk needs a score/reason trace.
- Keep the library provider-independent. Do not add an LLM call to the compression path.
- Prefer maintained, permissively licensed dependencies; record every model, dataset, and library in `docs/ACKNOWLEDGEMENTS.md`.
- Keep large data and model weights out of Git. Never commit keys or `.env` files.
- Before changing a shared contract, update `docs/contract.md` and notify affected owners.

## Authority
The human project lead decides architecture, shared interfaces, cross-lane dependencies, and scope. Agents act autonomously inside their lane.

## Lanes
| Agent | Owns |
|---|---|
| Engine | `ml/`, `data/` — compression pipeline, scorer adapters, evaluation |
| Product | `frontend/` — dashboard and API consumption |
| Platform | `backend/`, `docker-compose.yml` — HTTP boundary, configuration, packaging |
| Lead/shared | `AGENTS.md`, `README.md`, `scripts/`, `docs/` — scope, contracts, demo |

Do not edit outside your lane; request changes from the owner via the lead. `docs/` is open for new files.

## Interfaces
- `docs/contract.md` is the single source of truth for API routes, schemas, and the engine interface.
- The engine exposes only `ml/src/inference.py`; the backend does not import other `ml/` internals.
- Frontend talks to Backend only over the HTTP API.
- Contract changes: propose to the lead, commit `contract.md` alone, and notify affected agents before dependent work merges.
- Until the contract is filled, stub with mocks. Do not block.

## Data
- `data/raw/` is immutable. Derived data goes in `data/processed/`.
- `ml/data/` and `ml/models/` hold derived artifacts only.
- Large files, datasets, and weights stay out of Git. Document how to obtain them in `data/README.md`.

## Dependencies
Keep them necessary and justified. Add each only to its own lane's manifest. A new external API, hosted model, or service needs one proposal to the lead (need, alternatives, cost); continue with the local fallback until answered. A local LLM is allowed only for genuinely language-heavy needs.

## Evaluation
`ml/src/evaluation.py` compares scorers at a fixed token budget. Results go in `docs/evaluation.md`; do not claim quality preservation without recorded evidence.

## Conventions
- Branches: `<lane>/<topic>`. PR to `main`, merged by the lead. No force-push.
- `main` must always run with the command above.
- Secrets, `.env`, and machine-specific config stay out of Git.
- Log non-obvious decisions in `docs/decisions.md` (one line: decision, reason).
