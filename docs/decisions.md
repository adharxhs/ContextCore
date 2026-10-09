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
2026-10-09 | Near-dedup uses RapidFuzz fuzz.ratio when available (0.88 threshold default), falls back to Jaccard similarity on import failure | fuzz.ratio is character-based and deterministic; Jaccard is pure offline with no external deps; both are explainable
2026-10-09 | Protected spans tracked with byte-level positions and reason labels in ProtectionMatch for future narrowing | enables fine-grained protection and auditing; chunk-level boolean is retained for backward compatibility
2026-10-09 | Negation pattern excludes bare "no"/"nor" to avoid false positives while retaining critical negations like "not", "never", "cannot" | prevents false positives on common words; maintains protection against critical negations; tested with `test_unprotected_plain_prose`
2026-10-09 | Add `budget_exceeded: bool` field to CompressionResult for machine-readable over-budget detection | engine owner can override; adds explicit signal for budget overflow; saved_tokens clamped to non-negative; contract updated
2026-10-09 | Bidirectional QA dependency retention: assistant answers pull user questions, and user questions pull subsequent assistant answers | improves coherence of question-answer pairs; only includes assistant answers with score > 0.3; tested with `test_qa_dependency_retention`
