# ContextCore

ContextCore is an explainable, query-aware context-compression layer for chat LLM requests. It selects relevant evidence under a token budget **without** using a separate generative LLM call to summarize every request. Given a system prompt, chat history, external context blocks, and a query, it returns shorter query-relevant context plus a per-chunk provenance trace.

The hackathon MVP comprises a Python compression library, a thin FastAPI wrapper, a comparison dashboard, and a reproducible 40-case benchmark. One real chat model is used only for the optional full-context vs compressed-context answer comparison.

## Run

```
docker compose up --build
```

Backend: http://localhost:8000 (health: `/health`, docs: `/docs`, compression: `POST /v1/compress`).
Frontend dashboard: http://localhost:5173.

Without Docker, from the repo root:

```
pip install -r backend/requirements.txt
pip install tiktoken rank-bm25 numpy fastembed
uvicorn backend.app.main:app --reload
```

`fastembed` provides the dense embedding and cross-encoder ONNX models (`BAAI/bge-small-en-v1.5`, `BAAI/bge-reranker-base`); the first run downloads them. The `bm25` scorer needs only `rank-bm25`. If the engine cannot be imported, the backend returns a clearly-scoped offline fallback so the dashboard stays usable.

Copy `.env.example` to `.env` only if environment overrides are needed. Provider credentials are evaluation-only and are never required to run compression.

## Evaluate

```
python scripts/evaluate.py --scorers bm25,dense,hybrid,cross_encoder --require-tokenizer
```

Runs the 40 version-controlled cases in `data/benchmark/cases.jsonl` through the real engine
(`ml.src.inference.compress_context`) at each case's token budget and reports evidence recall,
token reduction, p50/p95 latency, and budget overruns. The harness records the **verified** engine
tokenizer (`tiktoken cl100k_base`); `--require-tokenizer` fails the run if the engine silently used
its heuristic fallback. `--mock` exercises only the harness pipeline and must never be reported as
a result.

Run the test suites and contract gate:

```
python -m pytest
python -m ruff check scripts
python scripts/check_contract.py
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

Real engine benchmark results (see `docs/evaluation.md`) currently **fail the budget acceptance
check**: on 40/40 cases the compressed output exceeds the token budget and mean token reduction is
~7%. Root cause is over-broad protection in the engine (87% of chunks classified protected, so the
protected content alone exceeds every budget) and is assigned to the Engine owner. The harness
detects and reports these overruns rather than hiding them. Do not cite the mock pipeline numbers
(92.5% recall / 54% reduction) as engine performance.

## Layout

| Path | Purpose |
|---|---|
| `backend/` | FastAPI app, services, tests |
| `frontend/` | UI and demo dashboard |
| `ml/` | Compression engine, scorer adapters (Engine lane) |
| `data/` | Benchmark cases and derived results (see `data/README.md`) |
| `docs/` | Contract, decisions, evaluation, acknowledgements |
| `scripts/` | Evaluation harness, contract gate, utilities |

Contributor and agent rules: see `AGENTS.md`. Shared interface: `docs/contract.md`.
