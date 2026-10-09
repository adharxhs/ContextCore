# Acknowledgements
Record every AI tool, library, model, and dataset used, with its license.

| Item | Type | License | Used for |
|---|---|---|---|
| `rank-bm25` | Library | Apache-2.0 | BM25 lexical relevance scoring |
| `fastembed` | Library | Apache-2.0 | Dense ONNX embeddings and cross-encoder reranking |
| `tiktoken` | Library | MIT | Token counting and boundary budgeting |
| `rapidfuzz` | Library | MIT | Near-deduplication string similarity |
| `BAAI/bge-small-en-v1.5` | Model | MIT | Dense embedding model |
| `BAAI/bge-reranker-base` | Model | MIT | Cross-encoder reranker model |
| Benchmark cases `data/benchmark/cases.jsonl` | dataset (original, synthetic) | MIT (repo) | 40 query-aware compression scenarios authored in-repo; not derived from external material |
| `scripts/evaluate.py` | tooling | MIT (repo) | Validation harness; Python standard library only, no external runtime deps |
