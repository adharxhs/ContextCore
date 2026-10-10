"""Short, reproducible ContextCore demo (Validation & Lead lane).

Runs a fixed request through the real engine and prints the budget selection, compressed
output, token statistics, and the selected/dropped provenance trace. Defaults to ``bm25``
so it runs without downloading embedding models.

Usage::

    python scripts/demo.py
    python scripts/demo.py --scorer hybrid --budget 80
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SYSTEM_PROMPT = (
    "You are a support assistant. Never invent account balances, dates, or policy terms."
)
HISTORY = [
    {"role": "user", "content": "How long will my refund take, and can I track it?"},
]
CONTEXT_BLOCKS = [
    {
        "id": "refund-policy",
        "content": (
            "Eligible refunds are issued to the original payment method within 5 business days."
        ),
        "source": "support-kb",
    },
    {
        "id": "tracking",
        "content": (
            "Customers receive an email confirmation and transaction reference once the "
            "refund is initiated."
        ),
        "source": "support-kb",
    },
    {
        "id": "office-news",
        "content": "The support lounge added plants, a coffee corner, and a reading shelf.",
        "source": "newsletter",
    },
    {
        "id": "brand-notes",
        "content": "The company palette uses deep green, warm cream, and a leaf motif.",
        "source": "brand-guide",
    },
    {
        "id": "order-cs-2048",
        "content": "Order #CS-2048 was marked delivered on 2026-10-02.",
        "source": "orders",
    },
]
QUERY = "How long will my refund take, and can I track it?"


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _state(trace: Any) -> str:
    if _get(trace, "protected"):
        return "PROTECTED"
    return "KEPT" if _get(trace, "selected") else "DROPPED"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scorer", default="bm25", help="bm25, dense, hybrid, or cross_encoder")
    parser.add_argument("--budget", type=int, default=60, help="token budget")
    args = parser.parse_args(argv)

    from ml.src.inference import compress_context

    result = compress_context(
        system_prompt=SYSTEM_PROMPT,
        history=HISTORY,
        context_blocks=CONTEXT_BLOCKS,
        query=QUERY,
        token_budget=args.budget,
        scorer=args.scorer,
    )

    input_tokens = _get(result, "input_tokens", 0)
    output_tokens = _get(result, "output_tokens", 0)
    saved = _get(result, "saved_tokens", 0)
    reduction = (saved / input_tokens * 100) if input_tokens else 0.0

    print(f"scorer={args.scorer}  budget={args.budget}")
    print(
        f"input_tokens={input_tokens}  output_tokens={output_tokens}  saved={saved} "
        f"({reduction:.1f}% reduction)  over_budget={output_tokens > args.budget}"
    )
    print(f"compression_ms={_get(result, 'compression_ms', 0)}")
    print("\n--- compressed_text ---")
    print(_get(result, "compressed_text", ""))

    print("\n--- trace (original order) ---")
    traces = list(_get(result, "selected_chunks", [])) + list(_get(result, "dropped_chunks", []))
    traces.sort(key=lambda t: _get(t, "original_index", 0))
    for trace in traces:
        print(
            f"[{_state(trace):9}] {_get(trace, 'id'):18} {_get(trace, 'token_count'):>4}t "
            f"score={_get(trace, 'score'):<6} {_get(trace, 'reason')}"
        )

    over_budget = output_tokens > args.budget
    note = (
        "NOTE: over_budget=True means protected content exceeded the budget "
        "(known engine defect E1)."
    )
    print(f"\n{note if over_budget else 'Within budget.'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
