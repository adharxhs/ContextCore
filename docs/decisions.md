# Decisions
One line each: `time | decision | reason`

`2026-10-09 | data/benchmark/cases.jsonl is the engine-agnostic fixture location (not data/raw or data/processed) | cases are small, synthetic, version-controlled and must be lint-validated in CI`
`2026-10-09 | token_budget derived as max(0.5*input, evidence+12, 24) per case | guarantees compression is forced while recall stays feasible; engine must not exceed it`
2026-10-09 | Use FastEmbed ONNX runtime for dense and cross-encoder scoring | Fast offline inference without large PyTorch runtime dependencies
2026-10-09 | Retire the working name "Context Surgeon"; the project is ContextCore everywhere | one name across contract, README, AGENTS, and harness avoids interface confusion
2026-10-09 | Evaluation records the verified engine tokenizer, not the nominal label | the engine silently falls back to a heuristic if tiktoken fails, which would falsify token counts and reduction
2026-10-09 | Real benchmark results recorded honestly as failing the budget check (40/40 over budget) | acceptance must reflect the real engine; mock pipeline numbers must not be reported as performance
2026-10-09 | CI gates lint on validation-owned paths and runs all test suites plus scripts/check_contract.py | Engine/Product lint failures are known (defect P2) and must not be silently disabled repo-wide
2026-10-09 | Re-validate after engine fix 956f322 and record the update: reduction ~61%, overruns 10/40, recall 82-85% | results must always match the merged code, not the branch that produced them
