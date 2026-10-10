# Acknowledgements
This file records every model, library, dataset and AI tool used, with its license.

## AI tools

| Tool | Provider | Used for | Where |
|------|----------|----------|-------|
| Claude | Anthropic | Drafted the 16-case supplementary benchmark suite (2 cases per core category, repository schema) at the team's request; produced the evidence-position variants and the truncation-baseline script; helped write and update project documentation, including `docs/architecture.md` and this file. | `supplementary_benchmark/` (`cases_*.jsonl`, `results_*.json`, `make_cases.py`, `truncation_baselines.py`), `docs/architecture.md`, `docs/ACKNOWLEDGEMENTS.md` |

Notes on the Claude entry:
- Claude is not part of the compression path. ContextCore makes no generative LLM call when compressing.
- The supplementary suite was run once at HEAD `53a0228` with the BM25 scorer, and no case was edited after seeing results. The evidence-position sweep was added after seeing that head truncation scored 100% on the original suite.
- The cases were not independently human-labelled. See the limits stated in the submission report.
- The team reviewed and is responsible for all submitted content.

## Models

| Model | Used for | License |
|-------|----------|---------|
| BAAI/bge-small-en-v1.5 | Offline dense embeddings (dense and hybrid scorers) | MIT |
| BAAI/bge-reranker-base | Cross-encoder reranking | MIT |

## Libraries and tools

| Library / tool | Used for | License |
|----------------|----------|---------|
| Python 3.12 | Runtime | PSF License |
| FastAPI | HTTP API | MIT |
| Uvicorn | ASGI server | BSD-3-Clause |
| Pydantic v2 | Request/response validation | MIT |
| NumPy | In-house BM25 Okapi scoring | BSD-3-Clause |
| fastembed | Embedding and reranker runtime (ONNX) | Apache-2.0 |
| tiktoken (`cl100k_base`) | Token counting and budget control | MIT |
| rapidfuzz | Near-duplicate detection | MIT |
| scikit-learn | TF-IDF fallback scoring | BSD-3-Clause |
| nginx | Static dashboard server in Docker | BSD-2-Clause |
| Docker / Docker Compose | Packaging and local run | Apache-2.0 |
| pytest | Tests (79) | MIT |
| Ruff | Linting | MIT |
| GitHub Actions | CI | Service (GitHub terms) |

## Datasets

No external datasets were used. All 50 benchmark cases (40 core, 10 stress) are original synthetic fixtures authored in this repository. The 16-case supplementary suite and its position variants are original synthetic cases drafted with Claude.

## Verification note

Licenses above reflect the upstream projects' published licenses as generally known; confirm each against the installed package or model card before final submission.
