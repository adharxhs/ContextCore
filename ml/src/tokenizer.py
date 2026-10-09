import re
import tiktoken

_ENCODING_NAME = "cl100k_base"
_ENCODER = None


def get_encoder() -> tiktoken.Encoding:
    global _ENCODER
    if _ENCODER is None:
        try:
            _ENCODER = tiktoken.get_encoding(_ENCODING_NAME)
        except Exception:
            _ENCODER = None
    return _ENCODER


def count_tokens(text: str) -> int:
    if not text:
        return 0
    enc = get_encoder()
    if enc is not None:
        try:
            return len(enc.encode(text, disallowed_special=()))
        except Exception:
            pass
    # Fallback heuristic: ~4 chars per token or whitespace split
    words = re.findall(r"\w+|[^\w\s]", text, re.UNICODE)
    return max(1, len(words))


def encode_tokens(text: str) -> list[int]:
    if not text:
        return []
    enc = get_encoder()
    if enc is not None:
        try:
            return enc.encode(text, disallowed_special=())
        except Exception:
            pass
    return [hash(w) % 100000 for w in re.findall(r"\w+|[^\w\s]", text, re.UNICODE)]


def decode_tokens(tokens: list[int]) -> str:
    if not tokens:
        return ""
    enc = get_encoder()
    if enc is not None:
        try:
            return enc.decode(tokens)
        except Exception:
            pass
    return ""


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    if max_tokens <= 0:
        return ""
    enc = get_encoder()
    if enc is not None:
        try:
            tokens = enc.encode(text, disallowed_special=())
            if len(tokens) <= max_tokens:
                return text
            return enc.decode(tokens[:max_tokens])
        except Exception:
            pass
    words = text.split()
    return " ".join(words[:max_tokens])
