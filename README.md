# Context Surgeon

Context Surgeon is an explainable, query-aware context-compression layer for chat LLM requests. It selects relevant evidence under a token budget without using a separate generative LLM call to summarize every request.

The hackathon MVP comprises a Python compression library, a thin FastAPI wrapper, a comparison dashboard, and a reproducible 30–50-case benchmark. The demo uses one real chat model only to compare answers from original versus compressed context.

## Run
```
docker compose up --build
```
Backend: http://localhost:8000 (health: `/health`, docs: `/docs`). The compression endpoint is added after the interface is approved in `docs/contract.md`.

Without Docker, from the repo root:
```
pip install -r backend/requirements.txt -r ml/requirements.txt
uvicorn backend.app.main:app --reload
```

Copy `.env.example` to `.env` only if environment overrides are needed. Provider credentials are evaluation-only and must never be required to run compression.

## Layout
| Path | Purpose |
|---|---|
| `backend/` | FastAPI app, services, tests |
| `frontend/` | UI and demo |
| `ml/` | Compression engine, scorer adapters, evaluation |
| `data/` | Raw and processed datasets (see `data/README.md`) |
| `docs/` | Contract, decisions, evaluation, event notes |
| `scripts/` | Setup and utility scripts |

Contributor and agent rules: see `AGENTS.md`.
