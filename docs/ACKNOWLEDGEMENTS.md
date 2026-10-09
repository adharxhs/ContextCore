# Acknowledgements
Record every AI tool, library, model, and dataset used, with its license.

## Runtime libraries

| Item | Type | License | Used for |
|---|---|---|---|
| `rank-bm25` | Library | Apache-2.0 | BM25 lexical relevance scoring |
| `fastembed` | Library | Apache-2.0 | Dense ONNX embeddings and cross-encoder reranking |
| `onnxruntime` | Library | MIT | ONNX inference backend used by FastEmbed |
| `tiktoken` | Library | MIT | Token counting and boundary budgeting |
| `rapidfuzz` | Library | MIT | Near-deduplication string similarity (preferred path) |
| `numpy` | Library | BSD-3-Clause | Vector math in the scorers |
| `scikit-learn` | Library | BSD-3-Clause | TF-IDF cosine fallback for the dense scorer |
| `pydantic` | Library | MIT | Engine and API request/response models |
| `fastapi` | Library | MIT | HTTP boundary for `/v1/compress` |
| `uvicorn` | Library | BSD-3-Clause | ASGI server for the backend |
| `httpx` | Library | BSD-3-Clause | Backend API tests (test dependency only) |
| `pytest` | Library | MIT | Test runner (dev/test only) |
| `ruff` | Library | MIT | Lint gate (dev/test only) |

## Models

| Item | Type | License | Used for | Acquisition |
|---|---|---|---|---|
| `BAAI/bge-small-en-v1.5` (FastEmbed `Qdrant/bge-small-en-v1.5-onnx-Q`) | Model | MIT | Dense embedding scorer | Downloaded by FastEmbed on first use into its cache; see `data/README.md` |
| `BAAI/bge-reranker-base` | Model | MIT | Cross-encoder reranker | Downloaded by FastEmbed on first use into its cache; see `data/README.md` |

`ml/models/` is a mounted, git-ignored location for cached model weights. Model weights are not
committed; the BM25 scorer needs no model download at all.

## Data and tooling

| Item | Type | License | Used for |
|---|---|---|---|
| Benchmark cases `data/benchmark/cases.jsonl` | dataset (original, synthetic) | MIT (repo) | 50 query-aware compression scenarios authored in-repo; not derived from external material |
| `scripts/evaluate.py` | tooling | MIT (repo) | Validation harness; imports only the engine entry point and `tiktoken` via the engine |
| `scripts/check_contract.py` | tooling | MIT (repo) | Contract/interface/fixture consistency gate |
| `scripts/smoke.sh` | tooling | MIT (repo) | Docker health and `/v1/compress` smoke check |

## Optional evaluation adapter

The optional same-model full-vs-compressed answer comparison is **not configured or executed**. If
added, the chat model, provider, license, and fixed generation settings must be recorded here before
any answer-quality claim is made (see `docs/evaluation.md`).

## Engine manifest fix

Engine PR #8 (`d92b77b`) corrected `ml/requirements.txt` to pin the actually-imported runtime
dependencies (`fastembed`, `numpy`, `scikit-learn`, `rapidfuzz`, `tiktoken`, `pydantic`) and removed
the unused `sentence-transformers` entry.
