# Benchmark data

- `raw/`: immutable source fixtures or externally obtained material; never commit licensed/private
  material without permission. Currently empty — the benchmark is original and synthetic.
- `processed/`: derived benchmark artifacts (e.g. `scripts/evaluate.py` result JSON); not committed.
- `benchmark/cases.jsonl`: version-controlled benchmark cases run by `scripts/evaluate.py`. See
  `benchmark/README.md` for the schema, budget derivation, stress categories, and taxonomy.

The benchmark is authored in-repo; no external dataset is downloaded. Every `required_evidence`
string is grounded verbatim in the case's own content, so recall is well defined.

## Offline operation (BM25, no model download)

The `bm25` scorer needs only `tiktoken` and `rank-bm25`. Run the whole benchmark offline:

```
pip install tiktoken rank-bm25 numpy
python scripts/evaluate.py --scorers bm25 --require-tokenizer --fail-on-degraded
```

No model weight is downloaded and `detect_scorer_mode("bm25")` reports `n/a (lexical only)`.

## Optional dense / cross-encoder models

The `dense`, `hybrid`, and `cross_encoder` scorers use FastEmbed ONNX models
(`BAAI/bge-small-en-v1.5`, `BAAI/bge-reranker-base`), both MIT-licensed. FastEmbed downloads them
on first use. To keep weights out of Git and make runs reproducible, point the cache at a persistent
location:

```
# PowerShell (Windows)
$env:FASTEMBED_CACHE_PATH = "$PWD\ml\models\fastembed"
# bash (Linux/macOS)
export FASTEMBED_CACHE_PATH="$PWD/ml/models/fastembed"
```

Then run any scorer once to populate the cache. `docker-compose.yml` mounts `./ml/models` into
`/srv/ml/models`; set `FASTEMBED_CACHE_PATH=/srv/ml/models/fastembed` for the backend service to
reuse the same cache and avoid re-downloading on every container start.

If FastEmbed or the cache is unavailable, the scorers silently fall back to TF-IDF (`dense`) or
hybrid ranking (`cross_encoder`). The harness reports this as `degraded` and
`--fail-on-degraded` fails the run, so a degraded run is never presented as a model-backed result.

For every external dataset, document source, license, version, download steps, and permitted use in
`docs/ACKNOWLEDGEMENTS.md`.
