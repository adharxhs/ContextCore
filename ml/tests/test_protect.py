from ml.src.protect import detect_protected_spans


def test_protect_system_prompt():
    match = detect_protected_spans("You are a helpful assistant", source_type="system")
    assert match.is_protected
    assert any("system instructions" in r for r in match.reasons)


def test_protect_recent_user_turn():
    match = detect_protected_spans("What is the capital of France?", source_type="history", is_recent_user_turn=True)
    assert match.is_protected
    assert any("recent user turn" in r for r in match.reasons)


def test_protect_code():
    code_text = "def calculate_tax(amount: float) -> float:\n    return amount * 0.2"
    match = detect_protected_spans(code_text, source_type="context")
    assert match.is_protected
    assert any("code" in r for r in match.reasons)


def test_protect_entities_and_negations():
    text = "The user with id: usr_9929 did not receive the payout on 2026-03-01 of $500.00."
    match = detect_protected_spans(text, source_type="context")
    assert match.is_protected
    assert any("identifier" in r for r in match.reasons)
    assert any("date" in r for r in match.reasons)
    assert any("negation" in r for r in match.reasons)
    assert any("numeric" in r for r in match.reasons)
