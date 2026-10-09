"""Benchmark harness for ContextCore (lane 3: Validation & Lead).

Runs the version-controlled case set in ``data/benchmark/cases.jsonl`` through the
engine at each case's fixed token budget, once per requested scorer, and reports:

* evidence recall (required evidence that survived compression), split into the
  stable core taxonomy and the stress categories (near-dedup, oversized sentence,
  protected-only overflow, protected-span precision),
* input-token reduction,
* p50 / p95 compression latency,
* budget overruns, classified by whether protected content caused them,
* model mode for dense/cross-encoder scorers (ONNX model loaded vs silent fallback),
* failures (exceptions, lost evidence).

The harness never makes a generative LLM call. It imports only the engine's public
interface, ``ml.src.inference.compress_context`` (see ``docs/contract.md``). It also
verifies which tokenizer the engine actually used (tiktoken ``cl100k_base`` versus the
heuristic fallback) so recorded token counts are trustworthy. When the engine interface is
absent, ``--mock`` provides a deterministic keyword selector so the pipeline can be
exercised; mock output is clearly labelled and must not be reported as a result.

Usage::

    python scripts/evaluate.py --scorers bm25,dense,hybrid,cross_encoder
    python scripts/evaluate.py --mock --output data/processed/eval-mock.json
    python scripts/evaluate.py --scorers bm25 --require-tokenizer --fail-on-degraded
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

# Original eight taxonomy buckets form the stable "core" benchmark. The stress
# categories were added by the Validation lane to cover known failure modes; they
# are reported separately so they do not silently move the core trend line.
CORE_CATEGORIES = (
    "planted_fact",
    "redundant",
    "long_history",
    "code_ids_numbers",
    "structured_data",
    "negations",
    "qa_dependency",
    "distractors",
)
STRESS_CATEGORIES = (
    "near_dedup",
    "oversized_sentence",
    "protected_overflow",
    "protected_precision",
)

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


def verify_tokenizer() -> tuple[bool, str | None, str]:
    """Confirm the engine uses tiktoken ``cl100k_base`` and not the heuristic fallback.

    ``input_tokens``/``output_tokens`` and therefore every reduction figure depend on
    the engine tokenizer. If tiktoken cannot load, the engine silently falls back to a
    character/word heuristic, so a recorded "cl100k_base" label would be wrong. The
    benchmark records the *verified* encoder, not the nominal one.
    """
    repo_root = str(REPO_ROOT)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    try:
        from ml.src.tokenizer import get_encoder  # type: ignore
    except Exception as exc:  # pragma: no cover - environment dependent
        return False, None, f"could not import ml.src.tokenizer ({exc!r})"
    try:
        encoder = get_encoder()
    except Exception as exc:  # pragma: no cover - environment dependent
        return False, None, f"get_encoder() failed ({exc!r})"
    if encoder is None:
        return False, None, "tiktoken unavailable; engine used the heuristic fallback"
    name = getattr(encoder, "name", None)
    if name == "cl100k_base":
        return True, name, "tiktoken cl100k_base active"
    return False, name, f"engine tokenizer is {name!r}, expected 'cl100k_base'"


def detect_scorer_mode(scorer: str) -> str:
    """Report whether a dense/reranking scorer loaded its ONNX model or degraded.

    The engine's dense and cross-encoder scorers silently fall back to TF-IDF and
    hybrid ranking when FastEmbed or the model cache is unavailable. That changes
    what is being measured, so the run records ``full``, ``degraded``, or ``unknown``
    rather than reporting degraded numbers as model-backed results.
    """
    if scorer == "bm25":
        return "n/a (lexical only)"
    try:
        if scorer in ("dense", "hybrid"):
            from ml.src.scorers import dense as dense_mod  # type: ignore

            value = dense_mod._EMBED_MODEL
            reason = dense_mod._EMBED_MODEL_FAILURE_REASON
        elif scorer == "cross_encoder":
            from ml.src.scorers import cross_encoder as ce_mod  # type: ignore

            value = ce_mod._CROSS_ENCODER_MODEL
            reason = ce_mod._CROSS_ENCODER_FAILURE_REASON
        else:  # pragma: no cover - SCORERS is fixed
            return "unknown"
    except Exception:  # pragma: no cover - environment dependent
        return "unknown"
    if value is None:
        return "unknown (scorer not exercised)"
    if value is False:
        suffix = f": {reason}" if reason else " (model unavailable, fell back)"
        return f"degraded{suffix}"
    return "full (onnx model loaded)"


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

    explicit_over = _get(result, "budget_exceeded", None)
    if explicit_over is None:
        explicit_over = _get(result, "over_budget", None)  # retired legacy name
    over_budget = (
        bool(explicit_over)
        if explicit_over is not None
        else output_tokens > case["token_budget"]
    )
    over_budget_source = "engine (budget_exceeded)" if explicit_over is not None else "computed"

    selected_traces = _get(result, "selected_chunks", None) or []
    dropped_traces = _get(result, "dropped_chunks", None) or []
    protected_selected_tokens = sum(
        int(_get(c, "token_count", 0) or 0) for c in selected_traces if _get(c, "protected", False)
    )
    duplicates_dropped = sum(
        1 for c in dropped_traces if "duplicate" in str(_get(c, "reason", "")).lower()
    )

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
            "over_budget": over_budget,
            "over_budget_source": over_budget_source,
            "over_budget_protected": bool(over_budget and protected_selected_tokens > 0),
            "protected_selected_tokens": protected_selected_tokens,
            "duplicates_dropped": duplicates_dropped,
            "selected": _chunk_count(_get(result, "selected_chunks")),
            "dropped": _chunk_count(_get(result, "dropped_chunks")),
        }
    )
    return record


def summarize(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-scorer metrics, including a core/stress and per-category split."""
    summary: dict[str, Any] = {}
    scorers = sorted({r["scorer"] for r in records})
    for scorer in scorers:
        rows = [r for r in records if r["scorer"] == scorer]
        ok = [r for r in rows if r["ok"]]
        total_input = sum(r["input_tokens"] for r in ok)
        total_output = sum(r["output_tokens"] for r in ok)
        latencies = [r["compression_ms"] for r in ok]
        recalls = [r["evidence_recall"] for r in ok]

        by_category: dict[str, dict[str, Any]] = {}
        for row in ok:
            category = row.get("category", "unknown")
            bucket = by_category.setdefault(
                category, {"cases": 0, "full_recall": 0, "recall_sum": 0.0, "over_budget": 0}
            )
            bucket["cases"] += 1
            bucket["full_recall"] += 1 if row.get("evidence_recall") == 1.0 else 0
            bucket["recall_sum"] += row.get("evidence_recall", 0.0)
            bucket["over_budget"] += 1 if row.get("over_budget") else 0
        for bucket in by_category.values():
            bucket["mean_recall"] = (
                round(bucket["recall_sum"] / bucket["cases"], 4) if bucket["cases"] else 0.0
            )

        core = [r for r in ok if r.get("category") in CORE_CATEGORIES]
        stress = [r for r in ok if r.get("category") in STRESS_CATEGORIES]

        summary[scorer] = {
            "cases": len(rows),
            "errors": len(rows) - len(ok),
            "mean_evidence_recall": round(sum(recalls) / len(recalls), 4) if recalls else 0.0,
            "core_mean_evidence_recall": round(
                sum(r["evidence_recall"] for r in core) / len(core), 4
            ) if core else 0.0,
            "stress_mean_evidence_recall": round(
                sum(r["evidence_recall"] for r in stress) / len(stress), 4
            ) if stress else 0.0,
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
            "over_budget_protected_cases": sum(1 for r in ok if r.get("over_budget_protected")),
            "budget_overrun_rate": round(
                sum(1 for r in ok if r.get("over_budget")) / len(ok), 4
            ) if ok else 0.0,
            "mean_overrun_tokens": round(
                sum(max(0, r["output_tokens"] - r.get("budget", r["output_tokens"])) for r in ok)
                / len(ok),
                1,
            ) if ok else 0.0,
            "missed_evidence_cases": sum(1 for r in ok if r.get("missed_evidence")),
            "by_category": by_category,
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


def print_summary(
    summary: dict[str, Any],
    engine_mode: str,
    tokenizer: str,
    scorer_modes: dict[str, str] | None = None,
) -> None:
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
    if scorer_modes:
        for scorer, mode in scorer_modes.items():
            print(f"  model mode [{scorer}]: {mode}")
        print()
    for scorer, s in summary.items():
        print(
            f"  recall [{scorer}]: core {_fmt_pct(s['core_mean_evidence_recall'])}"
            f" | stress {_fmt_pct(s['stress_mean_evidence_recall'])}"
            f" | all {_fmt_pct(s['mean_evidence_recall'])}"
        )
    print()
    for scorer, s in summary.items():
        cats = s.get("by_category", {})
        line = "  ".join(
            f"{c}:{_fmt_pct(cats[c]['mean_recall'])}" for c in STRESS_CATEGORIES if c in cats
        )
        if line:
            print(f"  stress categories [{scorer}]: {line}")
    print()
    for scorer, s in summary.items():
        if s.get("over_budget_cases"):
            print(
                f"  ! {scorer}: budget overrun on {s['over_budget_cases']}/{s['cases']} cases "
                f"({_fmt_pct(s['budget_overrun_rate'])}), mean +{s['mean_overrun_tokens']} tokens, "
                f"protected-linked {s['over_budget_protected_cases']}/{s['over_budget_cases']}"
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
    parser.add_argument(
        "--require-tokenizer",
        action="store_true",
        help="exit 1 unless the engine tokenizer is verified as tiktoken cl100k_base",
    )
    parser.add_argument(
        "--fail-on-degraded",
        action="store_true",
        help="exit 1 if a dense/cross-encoder scorer silently degraded to a fallback ranker",
    )
    parser.add_argument(
        "--fail-on-over-budget",
        action="store_true",
        help="exit 1 if any case output exceeds its token budget",
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
        tokenizer_verified, actual_tokenizer, tokenizer_detail = (
            False,
            None,
            "mock mode: engine tokenizer not exercised",
        )
        print(
            "!! MOCK MODE: numbers are placeholders and must not be reported as results.",
            file=sys.stderr,
        )
    else:
        tokenizer_verified, actual_tokenizer, tokenizer_detail = verify_tokenizer()
        print(f"Tokenizer: {tokenizer_detail}")
        if not tokenizer_verified:
            print(
                "!! WARNING: engine tokenizer is not tiktoken cl100k_base; token counts and "
                "reduction figures use the fallback heuristic.",
                file=sys.stderr,
            )

    records: list[dict[str, Any]] = []
    for scorer in scorers:
        for case in cases:
            records.append(run_case(case, scorer, compressor))

    scorer_modes = {} if args.mock else {scorer: detect_scorer_mode(scorer) for scorer in scorers}
    summary = summarize(records)
    print_summary(summary, engine_mode, args.tokenizer, scorer_modes)

    execution_mode = "mock" if args.mock else "engine"
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "engine": engine_mode,
        "execution_mode": execution_mode,
        "tokenizer": args.tokenizer,
        "tokenizer_verified": tokenizer_verified,
        "tokenizer_actual": actual_tokenizer,
        "tokenizer_detail": tokenizer_detail,
        "scorer_modes": scorer_modes,
        "cases_file": str(args.cases),
        "case_count": len(cases),
        "core_categories": list(CORE_CATEGORIES),
        "stress_categories": list(STRESS_CATEGORIES),
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

    if args.require_tokenizer and not tokenizer_verified:
        print(
            f"FAIL: engine tokenizer not verified (nominal label {args.tokenizer!r})",
            file=sys.stderr,
        )
        return 1

    if args.min_recall is not None:
        for scorer, s in summary.items():
            if s["mean_evidence_recall"] < args.min_recall:
                print(
                    f"FAIL {scorer}: mean recall {s['mean_evidence_recall']:.4f} "
                    f"< {args.min_recall:.4f}",
                    file=sys.stderr,
                )
                return 1

    if args.fail_on_degraded:
        degraded = [s for s, mode in scorer_modes.items() if mode.startswith("degraded")]
        unknown = [s for s, mode in scorer_modes.items() if mode.startswith("unknown")]
        if degraded or unknown:
            print(
                f"FAIL: scorer model mode is not full (degraded={degraded}, unknown={unknown}); "
                "model-backed numbers cannot be claimed.",
                file=sys.stderr,
            )
            return 1

    if args.fail_on_over_budget:
        over = sum(s["over_budget_cases"] for s in summary.values())
        if over:
            print(f"FAIL: {over} case(s) exceeded the token budget", file=sys.stderr)
            return 1

    total_errors = sum(s["errors"] for s in summary.values())
    return 1 if total_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
