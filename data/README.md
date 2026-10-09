# Benchmark data

- `raw/`: immutable source fixtures or externally obtained material; never commit licensed/private material without permission.
- `processed/`: derived benchmark artifacts (e.g. `scripts/evaluate.py` result JSON); not committed.
- `benchmark/cases.jsonl`: version-controlled benchmark cases run by `scripts/evaluate.py`. See `benchmark/README.md` for the schema, budget derivation, and taxonomy.

For every external dataset, document source, license, version, download steps, and permitted use in `docs/ACKNOWLEDGEMENTS.md`.
