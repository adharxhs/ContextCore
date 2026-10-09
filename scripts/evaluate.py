"""Benchmark harness for Context Surgeon (lane 3: Validation & Lead).

Runs the version-controlled case set in ``data/benchmark/cases.jsonl`` through the
engine at each case's fixed token budget, once per requested scorer, and reports:

* evidence recall (required evidence that survived compression),
* input-token reduction,
* p50 / p95 compression latency,
* failures (exceptions, budget overruns, lost evidence).

The harness never makes a generative LLM call. It imports only the engine's public
interface, ``ml.src.inference.compress_context`` (see ``docs/contract.md``). Until the
engine lands, ``--mock`` provides a deterministic keyword selector so the pipeline can be
exercised; mock output is clearly labelled and must not be reported as a result.

Usage::

    python scripts/evaluate.py --scorers bm25,dense,hybrid,cross_encoder
    python scripts/evaluate.py --mock --output data/processed/eval-mock.json
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASES = REPO_ROOT / "data" / "benchmark" / "cases.jsonl"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "processed" / "eval-results.json"

SCORERS = ("bm25", "dense", "hybrid", "cross_encoder")
DEFAULT_SCORERS = ("hybrid",)
# Must match the tokenizer configured in the engine; recorded with every result.
DEFAULT_TOKENIZER = "cl100k_base (tiktoken)"

CASE_FIELDS = (
    "id",
    "category",
    "system_prompt",
    "history",
    "context_blocks",
    "query",
    "token_budget",
    "required_evidence",
)


# --------------------------------------------------------------------------- #
# Case loading
# --------------------------------------------------------------------------- #
def load_cases(path: Path) -> list[dict[str, Any]]:
    """Load and validate JSONL benchmark cases."""
    if not path.exists():
        raise FileNotFoundError(f"case file not found: {path}")
    cases: list[dict[str, Any]] = []
    seen: set[str] = set()
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        try:
            case = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{lineno}: invalid JSON: {exc}") from exc
        missing = [f for f in CASE_FIELDS if f not in case]
        if missing:
            raise ValueError(f"{path}:{lineno}: case {case.get('id')!r} missing {missing}")
        if case["id"] in seen:
            raise ValueError(f"{path}:{lineno}: duplicate case id {case['id']!r}")
        seen.add(case["id"])
        if not isinstance(case["token_budget"], int) or case["token_budget"] <= 0:
            raise ValueError(f"{path}:{lineno}: token_budget must be a positive int")
        if not isinstance(case["required_evidence"], list):
            raise ValueError(f"{path}:{lineno}: required_evidence must be a list")
        cases.append(case)
    if not cases:
        raise ValueError(f"{path}: no cases found")
    return cases


# --------------------------------------------------------------------------- #
# Engine loading
# --------------------------------------------------------------------------- #
def load_engine() -> Callable[..., Any]:
    """Import the public engine entry point defined in the contract."""
    repo_root = str(REPO_ROOT)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    try:
        from ml.src.inference import compress_context  # type: ignore
    except Exception as exc:  # pragma: no cover - environment dependent
        raise SystemExit(
            "could not import ml.src.inference.compress_context. The engine (lane 1) has not "
            f"exposed the contract interface yet ({exc!r}). Re-run with --mock to exercise the "
            "harness only."
        ) from exc
    return compress_context


def _get(obj: Any, key: str, default: Any = None) -> Any:
    """Read a field from a dataclass, dict, or pydantic model."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _chunk_count(value: Any) -> int:
    if value is None:
        return 0
    try:
        return len(value)
    except TypeError:
        return 0


# --------------------------------------------------------------------------- #
# Evidence matching
# --------------------------------------------------------------------------- #
def normalize(text: str) -> str:
    """Casefold and collapse whitespace so evidence matching ignores formatting."""
    return re.sub(r"\s+", " ", str(text)).strip().lower()


def evidence_recall(compressed_text: str, required: Sequence[str]) -> tuple[float, list[str]]:
    """Fraction of required evidence present verbatim (normalized) in the output."""
    if not required:
        return 1.0, []
    haystack = normalize(compressed_text)
    missed = [item for item in required if normalize(item) not in haystack]
    return (len(required) - len(missed)) / len(required), missed


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile (q in [0, 1]); returns 0.0 for empty input."""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    pos = (len(ordered) - 1) * q
    low = math.floor(pos)
    high = math.ceil(pos)
    if low == high:
        return float(ordered[low])
    return float(ordered[low] + (ordered[high] - ordered[low]) * (pos - low))


@dataclass
class RunOutcome:
    record: dict[str, Any]


def run_case(case: dict[str, Any], scorer: str, compressor: Callable[..., Any]) -> dict[str, Any]:
    """Compress one case and score it."""
    record: dict[str, Any] = {
        "case_id": case["id"],
        "category": case["category"],
        "scorer": scorer,
        "budget": case["token_budget"],
        "ok": False,
        "error": None,
    }
    started = time.perf_counter()
    try:
        result = compressor(
            system_prompt=case["system_prompt"],
            history=case["history"],
            context_blocks=case["context_blocks"],
            query=case["query"],
            token_budget=case["token_budget"],
            scorer=scorer,
        )
    except Exception as exc:  # noqa: BLE001 - record and continue
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["wall_ms"] = (time.perf_counter() - started) * 1000.0
        return record

    wall_ms = (time.perf_counter() - started) * 1000.0
    compressed_text = _get(result, "compressed_text", "") or ""
    input_tokens = int(_get(result, "input_tokens", 0) or 0)
    output_tokens = int(_get(result, "output_tokens", 0) or 0)
    compression_ms = _get(result, "compression_ms", None)
    if compression_ms is None:
        compression_ms = wall_ms

    recall, missed = evidence_recall(compressed_text, case["required_evidence"])
    reduction = (1.0 - output_tokens / input_tokens) if input_tokens else 0.0

    record.update(
        {
            "ok": True,
            "evidence_recall": round(recall, 4),
            "missed_evidence": missed,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "saved_tokens": int(_get(result, "saved_tokens", input_tokens - output_tokens) or 0),
            "reduction": round(reduction, 4),
            "compression_ms": round(float(compression_ms), 3),
            "wall_ms": round(wall_ms, 3),
            "over_budget": output_tokens > case["token_budget"],
            "selected": _chunk_count(_get(result, "selected_chunks")),
            "dropped": _chunk_count(_get(result, "dropped_chunks")),
        }
    )
    return record


def summarize(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-scorer metrics."""
    summary: dict[str, Any] = {}
    scorers = sorted({r["scorer"] for r in records})
    for scorer in scorers:
        rows = [r for r in records if r["scorer"] == scorer]
        ok = [r for r in rows if r["ok"]]
        total_input = sum(r["input_tokens"] for r in ok)
        total_output = sum(r["output_tokens"] for r in ok)
        latencies = [r["compression_ms"] for r in ok]
        recalls = [r["evidence_recall"] for r in ok]
        summary[scorer] = {
            "cases": len(rows),
            "errors": len(rows) - len(ok),
            "mean_evidence_recall": round(sum(recalls) / len(recalls), 4) if recalls else 0.0,
            "full_recall_cases": sum(1 for r in recalls if r == 1.0),
            "token_reduction_micro": round(
                1.0 - total_output / total_input, 4
            ) if total_input else 0.0,
            "token_reduction_macro": round(
                sum(r["reduction"] for r in ok) / len(ok), 4
            ) if ok else 0.0,
            "latency_p50_ms": round(percentile(latencies, 0.50), 3),
            "latency_p95_ms": round(percentile(latencies, 0.95), 3),
            "over_budget_cases": sum(1 for r in ok if r.get("over_budget")),
            "missed_evidence_cases": sum(1 for r in ok if r.get("missed_evidence")),
        }
    return summary


# --------------------------------------------------------------------------- #
# Mock engine (pipeline verification only)
# --------------------------------------------------------------------------- #
_WORD = re.compile(r"[a-z0-9]+")


def _tokenize_ws(text: str) -> list[str]:
    return text.split()


def _mock_compress(
    system_prompt: str,
    history: list[dict[str, Any]],
    context_blocks: list[dict[str, Any]],
    query: str,
    token_budget: int,
    scorer: str,
) -> dict[str, Any]:
    """Deterministic keyword-overlap selector. NOT a real scorer; pipeline testing only."""
    query_terms = {t for t in _WORD.findall(query.lower()) if len(t) > 2}

    chunks: list[dict[str, Any]] = []
    for i, m in enumerate(history):
        chunks.append({"kind": "history", "index": i, "text": str(m.get("content", ""))})
    for i, b in enumerate(context_blocks):
        chunks.append(
            {"kind": "context", "index": i, "text": str(b.get("content", "")), "id": b.get("id")}
        )

    def score(chunk: dict[str, Any]) -> int:
        return len(query_terms & set(_WORD.findall(chunk["text"].lower())))

    forced = [system_prompt]
    if chunks and chunks[-1]["kind"] == "history":
        forced.append(chunks[-1]["text"])
    remaining = [c for c in chunks if c["text"] not in forced]
    remaining.sort(key=lambda c: (-score(c), c["kind"], c["index"]))

    keep: list[dict[str, Any]] = []
    used = len(_tokenize_ws(" ".join(forced)))
    for chunk in remaining:
        cost = max(1, len(_tokenize_ws(chunk["text"])))
        if used + cost > token_budget:
            continue
        keep.append(chunk)
        used += cost

    keep_ids = {id(c) for c in keep}
    ordered = forced[:]
    ordered += [c["text"] for c in chunks if id(c) in keep_ids]
    compressed = "\n".join(ordered)
    input_tokens = len(_tokenize_ws(" ".join([system_prompt] + [c["text"] for c in chunks])))
    output_tokens = len(_tokenize_ws(compressed))
    kept = keep_ids | {
        id(c) for c in chunks if c["kind"] == "history" and c["text"] in forced
    }
    return {
        "compressed_text": compressed,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "saved_tokens": input_tokens - output_tokens,
        "compression_ms": 0.0,
        "selected_chunks": [
            {"id": c.get("id") or f"{c['kind']}:{c['index']}", "selected": True}
            for c in chunks
            if id(c) in kept
        ],
        "dropped_chunks": [
            {"id": c.get("id") or f"{c['kind']}:{c['index']}", "selected": False}
            for c in chunks
            if id(c) not in kept
        ],
    }


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def _fmt_pct(value: float) -> str:
    return f"{value * 100:5.1f}%"


def print_summary(summary: dict[str, Any], engine_mode: str, tokenizer: str) -> None:
    print()
    print(f"Engine: {engine_mode}   Tokenizer(nominal): {tokenizer}")
    header = (
        f"{'scorer':<14}{'cases':>6}{'err':>5}{'recall':>9}{'full':>6}"
        f"{'reduc(tok)':>12}{'p50 ms':>9}{'p95 ms':>9}{'over':>6}"
    )
    print(header)
    print("-" * len(header))
    for scorer, s in summary.items():
        print(
            f"{scorer:<14}{s['cases']:>6}{s['errors']:>5}{_fmt_pct(s['mean_evidence_recall']):>9}"
            f"{s['full_recall_cases']:>6}{_fmt_pct(s['token_reduction_micro']):>12}"
            f"{s['latency_p50_ms']:>9.3f}{s['latency_p95_ms']:>9.3f}{s['over_budget_cases']:>6}"
        )
    print()


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES, help="benchmark case JSONL")
    parser.add_argument(
        "--scorers", default=",".join(DEFAULT_SCORERS), help=f"comma list from {SCORERS}"
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="results JSON path")
    parser.add_argument(
        "--tokenizer", default=DEFAULT_TOKENIZER, help="tokenizer label recorded in results"
    )
    parser.add_argument("--limit", type=int, default=None, help="only run the first N cases")
    parser.add_argument("--budget", type=int, default=None, help="override every case token budget")
    parser.add_argument(
        "--mock", action="store_true", help="use the built-in mock selector (not a real result)"
    )
    parser.add_argument(
        "--min-recall", type=float, default=None, help="exit 1 if mean recall below this"
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    scorers = [s.strip() for s in args.scorers.split(",") if s.strip()]
    unknown = [s for s in scorers if s not in SCORERS]
    if unknown:
        raise SystemExit(f"unknown scorer(s) {unknown}; expected one of {SCORERS}")

    cases = load_cases(args.cases)
    if args.limit is not None:
        cases = cases[: args.limit]
    if args.budget is not None:
        if args.budget <= 0:
            raise SystemExit("--budget must be positive")
        for case in cases:
            case["token_budget"] = args.budget

    engine_mode = (
        "mock (pipeline verification only)" if args.mock else "ml.src.inference.compress_context"
    )
    compressor = _mock_compress if args.mock else load_engine()
    if args.mock:
        print(
            "!! MOCK MODE: numbers are placeholders and must not be reported as results.",
            file=sys.stderr,
        )

    records: list[dict[str, Any]] = []
    for scorer in scorers:
        for case in cases:
            records.append(run_case(case, scorer, compressor))

    summary = summarize(records)
    print_summary(summary, engine_mode, args.tokenizer)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "engine": engine_mode,
        "tokenizer": args.tokenizer,
        "cases_file": str(args.cases),
        "case_count": len(cases),
        "scorers": scorers,
        "budget_override": args.budget,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "summary": summary,
        "results": records,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"results written to {args.output}")

    if args.min_recall is not None:
        for scorer, s in summary.items():
            if s["mean_evidence_recall"] < args.min_recall:
                print(
                    f"FAIL {scorer}: mean recall {s['mean_evidence_recall']:.4f} "
                    f"< {args.min_recall:.4f}",
                    file=sys.stderr,
                )
                return 1

    total_errors = sum(s["errors"] for s in summary.values())
    return 1 if total_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
