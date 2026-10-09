"""Unit tests for the lane-3 evaluation harness (engine-independent)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.evaluate import (
    CASE_FIELDS,
    CORE_CATEGORIES,
    DEFAULT_CASES,
    STRESS_CATEGORIES,
    _mock_compress,
    detect_scorer_mode,
    evidence_recall,
    load_cases,
    normalize,
    percentile,
    run_case,
    summarize,
    verify_tokenizer,
)


def test_normalize_collapses_whitespace_and_case() -> None:
    assert normalize("  Hello\n\tWorld  ") == "hello world"


def test_evidence_recall_all_present() -> None:
    recall, missed = evidence_recall(
        "The limit is 1000 requests per minute.", ["1000 requests per minute"]
    )
    assert recall == 1.0
    assert missed == []


def test_evidence_recall_reports_missed() -> None:
    recall, missed = evidence_recall("Nothing relevant here.", ["ACCT-90817"])
    assert recall == 0.0
    assert missed == ["ACCT-90817"]


def test_evidence_recall_ignores_whitespace_differences() -> None:
    recall, _ = evidence_recall("freeze   writes at\n02:00 UTC", ["freeze writes at 02:00 UTC"])
    assert recall == 1.0


def test_evidence_recall_empty_required_is_vacuous() -> None:
    assert evidence_recall("anything", []) == (1.0, [])


def test_percentile_interpolates() -> None:
    assert percentile([10.0, 20.0], 0.5) == pytest.approx(15.0)
    assert percentile([], 0.5) == 0.0
    assert percentile([42.0], 0.95) == 42.0


def test_load_cases_validates_shape(tmp_path: Path) -> None:
    bad = tmp_path / "bad.jsonl"
    bad.write_text(json.dumps({"id": "x"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_cases(bad)


def test_load_cases_rejects_duplicate_ids(tmp_path: Path) -> None:
    case = {
        "id": "dup",
        "category": "test",
        "system_prompt": "s",
        "history": [],
        "context_blocks": [],
        "query": "q",
        "token_budget": 10,
        "required_evidence": [],
    }
    path = tmp_path / "dup.jsonl"
    path.write_text(json.dumps(case) + "\n" + json.dumps(case) + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_cases(path)


def test_committed_cases_are_valid_and_span_taxonomy() -> None:
    cases = load_cases(DEFAULT_CASES)
    assert 30 <= len(cases) <= 50
    for case in cases:
        assert set(CASE_FIELDS) <= set(case)
        assert case["token_budget"] > 0
    categories = {c["category"] for c in cases}
    assert categories >= {
        "planted_fact",
        "redundant",
        "long_history",
        "code_ids_numbers",
        "structured_data",
        "negations",
        "qa_dependency",
        "distractors",
        "near_dedup",
        "oversized_sentence",
        "protected_overflow",
        "protected_precision",
    }
    counts: dict[str, int] = {}
    for case in cases:
        counts[case["category"]] = counts.get(case["category"], 0) + 1
    for category, count in counts.items():
        minimum = 4 if category in CORE_CATEGORIES else 2
        assert count >= minimum, f"category {category!r} has only {count} cases (minimum {minimum})"


def test_engine_tokenizer_is_tiktoken_cl100k_base() -> None:
    verified, name, detail = verify_tokenizer()
    assert verified, detail
    assert name == "cl100k_base"


def test_required_evidence_is_non_empty() -> None:
    cases = load_cases(DEFAULT_CASES)
    for case in cases:
        assert case["required_evidence"], f"{case['id']} has no required evidence"


def test_committed_evidence_is_grounded_and_budget_forces_reduction() -> None:
    cases = load_cases(DEFAULT_CASES)
    for case in cases:
        pool = normalize(
            " ".join(
                [case["system_prompt"]]
                + [m["content"] for m in case["history"]]
                + [b["content"] for b in case["context_blocks"]]
            )
        )
        for evidence in case["required_evidence"]:
            assert normalize(evidence) in pool, (
                f"{case['id']}: required evidence not found in case source"
            )
        whitespace_tokens = len(pool.split())
        assert case["token_budget"] < whitespace_tokens, (
            f"{case['id']}: budget {case['token_budget']} does not force reduction "
            f"(input ~{whitespace_tokens} whitespace tokens)"
        )


def test_run_case_with_mock_records_metrics() -> None:
    case = {
        "id": "unit-1",
        "category": "planted_fact",
        "system_prompt": "Answer from context.",
        "history": [{"role": "user", "content": "What is the limit?"}],
        "context_blocks": [
            {"id": "a", "content": "The limit is 1000 requests per minute."},
            {"id": "b", "content": "Unrelated cafeteria opening hours."},
        ],
        "query": "What is the limit?",
        "token_budget": 50,
        "required_evidence": ["1000 requests per minute"],
    }
    record = run_case(case, "hybrid", _mock_compress)
    assert record["ok"] is True
    assert record["evidence_recall"] == 1.0
    assert record["input_tokens"] >= record["output_tokens"]


def test_summarize_rolls_up_by_scorer() -> None:
    records = [
        {"scorer": "hybrid", "ok": True, "evidence_recall": 1.0, "reduction": 0.5,
         "input_tokens": 100, "output_tokens": 50, "compression_ms": 10.0, "budget": 60,
         "over_budget": False, "missed_evidence": []},
        {"scorer": "hybrid", "ok": True, "evidence_recall": 0.5, "reduction": 0.4,
         "input_tokens": 100, "output_tokens": 60, "compression_ms": 20.0, "budget": 60,
         "over_budget": False, "missed_evidence": ["x"]},
    ]
    summary = summarize(records)["hybrid"]
    assert summary["cases"] == 2
    assert summary["errors"] == 0
    assert summary["mean_evidence_recall"] == pytest.approx(0.75)
    assert summary["full_recall_cases"] == 1
    assert summary["latency_p50_ms"] == pytest.approx(15.0)
    assert summary["budget_overrun_rate"] == 0.0


def test_run_case_detects_budget_overrun() -> None:
    case = {
        "id": "over-1",
        "category": "planted_fact",
        "system_prompt": "s",
        "history": [],
        "context_blocks": [],
        "query": "q",
        "token_budget": 10,
        "required_evidence": [],
    }

    def too_big(**_: object) -> dict:
        return {
            "compressed_text": "kept text that is far too long",
            "input_tokens": 50,
            "output_tokens": 40,
            "saved_tokens": 10,
            "compression_ms": 1.0,
            "selected_chunks": [],
            "dropped_chunks": [],
        }

    record = run_case(case, "bm25", too_big)
    assert record["ok"] is True
    assert record["over_budget"] is True


def test_summarize_reports_overrun_rate_and_tokens() -> None:
    records = [
        {"scorer": "bm25", "ok": True, "evidence_recall": 1.0, "reduction": 0.0,
         "input_tokens": 100, "output_tokens": 80, "compression_ms": 1.0, "budget": 50,
         "over_budget": True, "missed_evidence": []},
        {"scorer": "bm25", "ok": True, "evidence_recall": 1.0, "reduction": 0.0,
         "input_tokens": 100, "output_tokens": 40, "compression_ms": 1.0, "budget": 50,
         "over_budget": False, "missed_evidence": []},
    ]
    summary = summarize(records)["bm25"]
    assert summary["over_budget_cases"] == 1
    assert summary["budget_overrun_rate"] == pytest.approx(0.5)
    assert summary["mean_overrun_tokens"] == pytest.approx(15.0)


def test_detect_scorer_mode_bm25_is_lexical() -> None:
    assert detect_scorer_mode("bm25").startswith("n/a")


def test_run_case_prefers_engine_over_budget_field() -> None:
    case = {
        "id": "ov-2", "category": "protected_overflow", "system_prompt": "s",
        "history": [], "context_blocks": [], "query": "q",
        "token_budget": 50, "required_evidence": [],
    }

    def engine_result(**_: object) -> dict:
        return {
            "compressed_text": "kept",
            "input_tokens": 100,
            "output_tokens": 40,
            "saved_tokens": 60,
            "compression_ms": 1.0,
            "budget_exceeded": True,
            "selected_chunks": [
                {"id": "c", "protected": True, "token_count": 40,
                 "selected": True, "reason": "protected overflow"},
            ],
            "dropped_chunks": [],
        }

    record = run_case(case, "bm25", engine_result)
    assert record["over_budget"] is True
    assert record["over_budget_source"] == "engine (budget_exceeded)"
    assert record["over_budget_protected"] is True
    assert record["protected_selected_tokens"] == 40


def test_run_case_counts_duplicates_dropped() -> None:
    case = {
        "id": "dup-x", "category": "near_dedup", "system_prompt": "s",
        "history": [], "context_blocks": [], "query": "q",
        "token_budget": 50, "required_evidence": [],
    }

    def engine_result(**_: object) -> dict:
        return {
            "compressed_text": "kept",
            "input_tokens": 100, "output_tokens": 40, "saved_tokens": 60,
            "compression_ms": 1.0,
            "selected_chunks": [],
            "dropped_chunks": [
                {"id": "d1", "selected": False,
                 "reason": "near-duplicate of chunk 'n1' (sim 0.95)"},
                {"id": "d2", "selected": False, "reason": "token budget exhausted"},
            ],
        }

    record = run_case(case, "bm25", engine_result)
    assert record["duplicates_dropped"] == 1


def test_summarize_splits_core_and_stress() -> None:
    records = [
        {"scorer": "bm25", "ok": True, "category": "planted_fact", "evidence_recall": 1.0,
         "reduction": 0.5, "input_tokens": 100, "output_tokens": 50, "compression_ms": 1.0,
         "budget": 60, "over_budget": False, "missed_evidence": []},
        {"scorer": "bm25", "ok": True, "category": "protected_overflow", "evidence_recall": 0.5,
         "reduction": 0.2, "input_tokens": 100, "output_tokens": 80, "compression_ms": 1.0,
         "budget": 60, "over_budget": True, "missed_evidence": ["x"]},
    ]
    summary = summarize(records)["bm25"]
    assert summary["core_mean_evidence_recall"] == 1.0
    assert summary["stress_mean_evidence_recall"] == 0.5
    assert summary["by_category"]["planted_fact"]["mean_recall"] == 1.0
    assert summary["by_category"]["protected_overflow"]["over_budget"] == 1


def test_core_and_stress_categories_are_disjoint() -> None:
    assert set(CORE_CATEGORIES).isdisjoint(STRESS_CATEGORIES)
