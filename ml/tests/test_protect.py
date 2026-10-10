from ml.src.protect import detect_protected_spans


def test_protect_system_prompt():
    match = detect_protected_spans("You are a helpful assistant", source_type="system")
    assert match.is_protected
    assert any("system instructions" in r for r in match.reasons)


def test_protect_recent_user_turn():
    match = detect_protected_spans(
        "What is the capital of France?", source_type="history", is_recent_user_turn=True
    )
    assert match.is_protected
    assert any("recent user turn" in r for r in match.reasons)


def test_protect_code():
    code_text = "def calculate_tax(amount: float) -> float:\n    return amount * 0.2"
    match = detect_protected_spans(code_text, source_type="context")
    assert match.is_protected
    assert any("code" in r for r in match.reasons)

    ts_code = "export interface Invoice {\n  id: string;\n  amount: number;\n}"
    match_ts = detect_protected_spans(ts_code, source_type="context")
    assert match_ts.is_protected

    sql_code = "SELECT user_id, email FROM users WHERE active = 1"
    match_sql = detect_protected_spans(sql_code, source_type="context")
    assert match_sql.is_protected


def test_protect_entities_and_negations():
    text = "The user with id: usr_9929 did not receive the payout on 2026-03-01 of $500.00."
    match = detect_protected_spans(text, source_type="context")
    assert match.is_protected
    assert any("identifier" in r for r in match.reasons)
    assert any("date" in r for r in match.reasons)
    assert any("negation" in r for r in match.reasons)
    assert any("numeric" in r for r in match.reasons)


def test_protect_structured_data():
    table = "| Header 1 | Header 2 |\n|---|---|\n| Cell 1 | Cell 2 |"
    match_table = detect_protected_spans(table, source_type="context")
    assert match_table.is_protected
    assert any("structured" in r for r in match_table.reasons)

    json_data = '{\n  "service": "billing",\n  "replicas": 3\n}'
    match_json = detect_protected_spans(json_data, source_type="context")
    assert match_json.is_protected

    csv_data = "col1,col2,col3\nval1,val2,val3\nval4,val5,val6"
    match_csv = detect_protected_spans(csv_data, source_type="context")
    assert match_csv.is_protected


def test_protect_identifiers_and_dates():
    id_text = "Tracking ACCT-90817 and merch-stage-db cluster"
    match_id = detect_protected_spans(id_text, source_type="context")
    assert match_id.is_protected
    assert any("identifier" in r for r in match_id.reasons)

    date_text = "Cutover at 02:00 UTC on 15 March"
    match_date = detect_protected_spans(date_text, source_type="context")
    assert match_date.is_protected
    assert any("date" in r for r in match_date.reasons)


def test_unprotected_plain_prose():
    prose = "The marketing team announced a spring campaign for the brand refresh."
    match = detect_protected_spans(prose, source_type="context")
    assert not match.is_protected
    assert len(match.reasons) == 0
