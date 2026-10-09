# Hackathon execution notes

## Rules
1. Development within the 24-hour window.
2. Work must be original.
3. AI tools, open-source software, and frameworks allowed with acknowledgement and team understanding.

## Brief distilled
- Build a lightweight library or API that returns shorter query-relevant LLM context.
- Do not depend on an extra full LLM compression call.
- Demonstrate token reduction, latency, and answer-quality preservation.
- Evaluate on at least 30 representative cases, including code-heavy and question-answer content.

## Checkpoints
- 0–4h: contract, protected-span/chunking baseline, benchmark fixtures.
- 4–9h: BM25/dense baseline and cross-encoder shortlist reranking.
- 9–14h: token budget, provenance trace, API and dashboard wiring.
- 14–18h: evaluation and full-versus-compressed LLM comparison.
- 18–20h: feature freeze, demo rehearsal, submission buffer.

## Process
- Blocked > 30 min: tell the lead the blocker, what was tried, what is needed.
- Status format: Done / Doing / Blocked, one line each.
- Demo fallback: fixed inputs, precomputed outputs, offline run.
- Presentation: whole team.
