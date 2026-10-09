"""Contract consistency checks for CI (Validation & Lead lane).

Fails with exit code 1 when the documented contract, the engine's public interface,
the HTTP schemas, the benchmark fixtures, or the project naming drift apart. This is a
cheap, dependency-light gate meant to run on every push; it does not exercise model
inference beyond the tokenizer.

Usage::

    python scripts/check_contract.py
"""

from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

CONTRACT = REPO_ROOT / "docs" / "contract.md"
README = REPO_ROOT / "README.md"
AGENTS = REPO_ROOT / "AGENTS.md"
CASES = REPO_ROOT / "data" / "benchmark" / "cases.jsonl"

ENGINE_PARAMS = [
    "system_prompt",
    "history",
    "context_blocks",
    "query",
    "token_budget",
    "scorer",
]
REQUEST_FIELDS = {
    "system_prompt",
    "history",
    "context_blocks",
    "query",
    "token_budget",
    "scorer",
}
RESPONSE_FIELDS = {
    "compressed_text",
    "input_tokens",
    "output_tokens",
    "saved_tokens",
    "compression_ms",
    "selected_chunks",
    "dropped_chunks",
}
TRACE_FIELDS = {
    "id",
    "source_type",
    "source",
    "original_index",
    "text",
    "token_count",
    "selected",
    "protected",
    "score",
    "reason",
}
CASE_FIELDS = {
    "id",
    "category",
    "system_prompt",
    "history",
    "context_blocks",
    "query",
    "token_budget",
    "required_evidence",
}
TAXONOMY = {
    "planted_fact",
    "redundant",
    "long_history",
    "code_ids_numbers",
    "structured_data",
    "negations",
    "qa_dependency",
    "distractors",
}

failures: list[str] = []


def check(condition: bool, message: str) -> None:
    status = "ok  " if condition else "FAIL"
    print(f"[{status}] {message}")
    if not condition:
        failures.append(message)


def check_engine() -> None:
    try:
        from ml.src.inference import compress_context
    except Exception as exc:  # pragma: no cover - environment dependent
        check(False, f"engine import: {exc!r}")
        return
    params = list(inspect.signature(compress_context).parameters)
    check(params == ENGINE_PARAMS, f"engine signature == contract {ENGINE_PARAMS}: got {params}")


def check_http_schema() -> None:
    try:
        from backend.app.api.schemas import ChunkTrace, CompressionResponse, CompressRequest
    except Exception as exc:  # pragma: no cover - environment dependent
        check(False, f"backend schema import: {exc!r}")
        return
    request_fields = set(CompressRequest.model_fields)
    check(REQUEST_FIELDS <= request_fields, "CompressRequest covers all contract inputs")
    response_fields = set(CompressionResponse.model_fields)
    check(RESPONSE_FIELDS <= response_fields, "CompressionResponse covers all contract outputs")
    trace_fields = set(ChunkTrace.model_fields)
    check(TRACE_FIELDS <= trace_fields, "ChunkTrace covers all contract trace fields")


def check_tokenizer() -> None:
    try:
        from ml.src.tokenizer import get_encoder
    except Exception as exc:  # pragma: no cover - environment dependent
        check(False, f"tokenizer import: {exc!r}")
        return
    encoder = get_encoder()
    name = getattr(encoder, "name", None)
    check(name == "cl100k_base", f"engine tokenizer is tiktoken cl100k_base (got {name!r})")


def check_cases() -> None:
    if not CASES.exists():
        check(False, f"benchmark cases exist at {CASES}")
        return
    lines = CASES.read_text(encoding="utf-8").splitlines()
    cases = [json.loads(line) for line in lines if line.strip()]
    check(30 <= len(cases) <= 50, f"benchmark has 30-50 cases (got {len(cases)})")
    bad = [c.get("id") for c in cases if not CASE_FIELDS <= set(c)]
    check(not bad, f"every case declares the full schema (bad: {bad})")
    categories = {c["category"] for c in cases}
    missing = TAXONOMY - categories
    check(TAXONOMY <= categories, f"benchmark covers the taxonomy (missing: {missing})")


def check_docs() -> None:
    contract_text = CONTRACT.read_text(encoding="utf-8") if CONTRACT.exists() else ""
    check(CONTRACT.exists(), "docs/contract.md exists")
    check("ContextCore" in contract_text, "contract uses the ContextCore name")
    check("Context Surgeon" not in contract_text, "contract has no stale 'Context Surgeon' name")
    for path in (README, AGENTS):
        text = path.read_text(encoding="utf-8") if path.exists() else ""
        check(path.exists(), f"{path.name} exists")
        check("ContextCore" in text, f"{path.name} uses the ContextCore name")
        check("Context Surgeon" not in text, f"{path.name} has no stale 'Context Surgeon' name")


def main() -> int:
    check_engine()
    check_http_schema()
    check_tokenizer()
    check_cases()
    check_docs()
    print()
    if failures:
        print(f"FAIL: {len(failures)} contract check(s) failed")
        return 1
    print("PASS: contract, engine, schema, fixtures, and naming agree")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
