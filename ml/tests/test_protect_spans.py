from ml.src.protect import detect_protected_spans


def test_protect_spans_extraction():
    text = "User id: usr_9929 on 2026-03-01"
    match = detect_protected_spans(text, source_type="context")
    assert match.is_protected
    assert len(match.spans) > 0
    assert any(s.reason == "identifier/hash/key" for s in match.spans)
    assert any(s.reason == "date/time entity" for s in match.spans)


def test_protect_spans_system_prompt():
    text = "You are a helpful assistant"
    match = detect_protected_spans(text, source_type="system")
    assert match.is_protected
    assert len(match.spans) == 1
    assert match.spans[0].start == 0
    assert match.spans[0].end == len(text)
    assert match.spans[0].reason == "system instructions protected by default"


def test_protect_spans_recent_user_turn():
    text = "What is this?"
    match = detect_protected_spans(text, source_type="history", is_recent_user_turn=True)
    assert match.is_protected
    assert len(match.spans) == 1
    assert match.spans[0].start == 0
    assert match.spans[0].end == len(text)
    assert match.spans[0].reason == "recent user turn protected"


def test_protect_spans_code_block():
    code = "def foo():\n    return 42"
    match = detect_protected_spans(code, source_type="context")
    assert match.is_protected
    assert any(s.reason == "code or syntax block" for s in match.spans)


def test_protect_spans_multiple_numbers():
    text = "Price is $50 and quantity is 100 units at 25% discount"
    match = detect_protected_spans(text, source_type="context")
    assert match.is_protected
    assert len([s for s in match.spans if s.reason == "numeric value/metric"]) >= 1


def test_protect_spans_negation():
    text = "This will never happen and it should not occur"
    match = detect_protected_spans(text, source_type="context")
    assert match.is_protected
    assert any(s.reason == "critical negation" for s in match.spans)


def test_protect_spans_selective_disable():
    text = "def func() at 2026-03-01 for usr_123"
    match_code_only = detect_protected_spans(
        text,
        source_type="context",
        protect_code=True,
        protect_numbers=False,
        protect_dates=False,
        protect_ids=False,
    )
    assert any(s.reason == "code or syntax block" for s in match_code_only.spans)
    assert not any(s.reason == "date/time entity" for s in match_code_only.spans)


def test_protect_spans_no_protection():
    text = "The marketing team announced a spring campaign"
    match = detect_protected_spans(text, source_type="context")
    assert not match.is_protected
    assert len(match.spans) == 0
    assert len(match.reasons) == 0
