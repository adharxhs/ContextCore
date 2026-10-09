# Evaluation

## Benchmark
- 30–50 version-controlled scenarios: planted early facts, redundant boilerplate, long histories, code/IDs/numbers, structured data, and irrelevant distractors.
- Each case defines the query, token budget, and required evidence.

## Comparisons
- Scorers: BM25, dense, hybrid, cross-encoder reranked hybrid.
- Fixed-budget metrics: evidence recall, input-token reduction, p50/p95 compression latency.
- End-to-end check: one chat LLM answers the same query with original and compressed context; use a fixed model/configuration and record a small answer rubric.

## Results
Record command, dependency/model versions, hardware, dataset revision, and failures as well as wins.
