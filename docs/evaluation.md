# Evaluation

## Benchmark
- 40 version-controlled scenarios in `data/benchmark/cases.jsonl` (see `data/benchmark/README.md`): planted early facts, redundant boilerplate, long histories, code/IDs/numbers, structured data, negations, QA dependencies, and irrelevant distractors.
- Each case defines the query, a token budget that forces ~50% reduction, and `required_evidence` grounded in the case's own content.

## Evidence recall
A case passes when every `required_evidence` item is present in `compressed_text` after
whitespace-normalized, case-folded substring matching. Recall is the fraction retained per
case; the summary reports the mean and the number of full-recall cases per scorer.

## Harness
```
python scripts/evaluate.py --scorers bm25,dense,hybrid,cross_encoder
```
- Imports only the engine's public entry point `ml.src.inference.compress_context`; no generative LLM call.
- Fixed-budget metrics per scorer: mean evidence recall, full-recall cases, input-token reduction (micro over all tokens and macro per case), p50/p95 compression latency, budget overruns, and errors.
- Emits `data/processed/eval-results.json` with metadata (timestamp, engine mode, case file, dependency-free platform report, tokenizer label) plus per-case traces.
- Until the engine exposes the interface, use `python scripts/evaluate.py --mock` to exercise the pipeline. Mock output is placeholder numbers, is clearly flagged, and must not be reported as results.
- Exit code is non-zero when any case fails; `--min-recall <r>` fails if any scorer's mean recall is below `r`.

## Results
Record command, dependency/model versions, hardware, dataset revision, tokenizer used, and failures as well as wins. The tokenizer must be stated because `input_tokens`/`output_tokens` depend on it. Do not claim quality preservation without recorded evidence.

## End-to-end quality check
The optional same-model comparison (full versus compressed context) reuses the same case
set, query, and a fixed model/configuration, scoring answers against a small recorded
rubric. This is the only step that uses a real chat LLM; it is evaluation-only. See `data/processed/` outputs and this file for any reported numbers.