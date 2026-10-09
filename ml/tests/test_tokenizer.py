from ml.src.tokenizer import count_tokens, decode_tokens, encode_tokens, truncate_to_tokens


def test_count_tokens():
    text = "Hello world! This is a test."
    tokens = count_tokens(text)
    assert tokens > 0
    assert count_tokens("") == 0


def test_encode_decode_tokens():
    text = "Antigravity context compression engine"
    enc = encode_tokens(text)
    assert len(enc) > 0
    dec = decode_tokens(enc)
    assert dec == text


def test_truncate_to_tokens():
    text = "One two three four five six seven eight nine ten"
    truncated = truncate_to_tokens(text, max_tokens=4)
    assert count_tokens(truncated) <= 4
