# Decisions
One line each: `time | decision | reason`

`2026-10-09 | data/benchmark/cases.jsonl is the engine-agnostic fixture location (not data/raw or data/processed) | cases are small, synthetic, version-controlled and must be lint-validated in CI`
`2026-10-09 | token_budget derived as max(0.5*input, evidence+12, 24) per case | guarantees compression is forced while recall stays feasible; engine must not exceed it`
