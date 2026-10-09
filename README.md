# ContextCore

ContextCore is an explainable, query-aware context-compression layer for chat LLM requests. It selects relevant evidence under a token budget **without** using a separate generative LLM call to summarize every request. Given a system prompt, chat history, external context blocks, and a query, it returns shorter query-relevant context plus a per-chunk provenance trace.

The hackathon MVP comprises a Python compression library, a thin FastAPI wrapper, a comparison dashboard, and a reproducible 50-case benchmark. One real chat model is used only for the optional full-context vs compressed-context answer comparison.

## Run

```
docker compose up --build
```

Backend: http://localhost:8000 (health: `/health`, docs: `/docs`, compression: `POST /v1/compress`).
Frontend dashboard: http://localhost:5173.

Without Docker, from the repo root:

```
pip install -r backend/requirements.txt -r ml/requirements.txt
uvicorn backend.app.main:app --reload
```

`ml/requirements.txt` pins the engine's actual runtime dependencies. For a fully offline run with no
model download, select the `bm25` scorer (needs only `tiktoken`, `rank-bm25`, `numpy`). The `dense`
and `cross_encoder` scorers use FastEmbed ONNX models (`BAAI/bge-small-en-v1.5`,
`BAAI/bge-reranker-base`, both MIT); the first run downloads them. Set `FASTEMBED_CACHE_PATH` to a
persistent directory (see `data/README.md`) and `docker-compose.yml` mounts `./ml/models` so weights
stay out of Git. If FastEmbed is unavailable the scorers degrade to TF-IDF/hybrid ranking; the
harness reports that mode. If the engine cannot be imported, the backend returns a clearly-scoped
offline fallback (`X-ContextCore-Execution: fallback`) so the dashboard stays usable.

Copy `.env.example` to `.env` only if environment overrides are needed. Provider credentials are evaluation-only and are never required to run compression.

## Evaluate

```
python scripts/evaluate.py --scorers bm25,dense,hybrid,cross_encoder --require-tokenizer --fail-on-degraded
```

Runs the 50 version-controlled cases in `data/benchmark/cases.jsonl` (40 core + 10 stress) through the
real engine (`ml.src.inference.compress_context`) at each case's token budget. It reports evidence
recall (overall, core, stress, per category), token reduction, p50/p95 latency, the **verified**
engine tokenizer (`tiktoken cl100k_base`), the dense/cross-encoder **model mode** (full vs degraded),
and budget overruns (flagged as protected-linked). `--require-tokenizer` fails if the engine silently
used its heuristic tokenizer; `--fail-on-degraded` fails if a scorer fell back; `--fail-on-over-budget`
fails on any budget overrun. `--mock` exercises only the harness pipeline and must never be reported
as a result. For a fully offline run use `--scorers bm25`.

Run the test suites, lint, and contract gate:

```
python -m pytest
python -m ruff check scripts
python scripts/check_contract.py
```

Docker smoke check (builds the stack, waits for `/health`, posts a BM25 compression):

```
bash scripts/smoke.sh
```

## Demo

A fixed, reproducible end-to-end demo (no model download with the default scorer):

```
python scripts/demo.py                 # bm25, budget 60
python scripts/demo.py --scorer hybrid --budget 80
```

It prints the budget selection, compressed text, token statistics, and the ordered
selected/dropped provenance trace. The web dashboard (`docker compose up --build`, then
http://localhost:5173) presents the same result visually.

## Validation status

Real engine benchmark results (see `docs/evaluation.md`, base `145a25c`): the engine reduces
tokens by ~59% with 80-83% mean evidence recall (core 80-84%, stress 80%). 16/50 cases still exceed
the token budget, all protected-linked and signaled by the engine's explicit `budget_exceeded`
field. Open Engine issues: evidence is lost on QA-dependency/long-history cases (E6), unprotected
oversized sentences are dropped (E8), and irrelevant protected spans crowd out relevant evidence
(E9). Fixed in Engine PR #8: the manifest gap (E3) and the over-budget signal (E4). Remaining
contract gaps: Engine `token_budget`/`tokenizer`, Product `budget_exceeded`/`execution_mode` body
fields, plus Product lint (P2). Do not cite the mock pipeline numbers as engine performance.
Cross-lane blockers are tracked in `docs/integration-status.md`.

## Layout

| Path | Purpose |
|---|---|
| `backend/` | FastAPI app, services, tests |
| `frontend/` | UI and demo dashboard |
| `ml/` | Compression engine, scorer adapters (Engine lane) |
| `data/` | Benchmark cases and derived results (see `data/README.md`) |
| `docs/` | Contract, decisions, evaluation, acknowledgements, integration status |
| `scripts/` | Evaluation harness, contract gate, smoke test, utilities |

Contributor and agent rules: see `AGENTS.md`. Shared interface: `docs/contract.md`.
