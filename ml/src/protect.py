import re
from typing import NamedTuple


class ProtectionMatch(NamedTuple):
    is_protected: bool
    reasons: list[str]


CODE_BLOCK_PATTERN = re.compile(
    r"(```[\s\S]*?```|`[^`\n]+`|"
    r"(?:^|\n)\s*(?:export\s+)?(?:async\s+)?(?:def|class|function|interface|type|enum)\s+[a-zA-Z_$]\w*|"
    r"(?:^|\n)\s*(?:const|let|var)\s+[a-zA-Z_$]\w*\s*=|"
    r"(?:^|\n)\s*(?:import\s+[a-zA-Z_]\w*|from\s+[a-zA-Z_]\w*\s+import)|"
    r"(?:^|\n)\s*(?:return|yield)\s+[^\n]+|"
    r"Traceback\s+\(most recent call last\):|(?:[A-Za-z0-9_]+(?:Error|Exception)):\s+|"
    r"\bSELECT\s+[\s\S]+?\s+FROM\s+[a-zA-Z0-9_.]+\b)",
    re.IGNORECASE,
)

DATE_PATTERN = re.compile(
    r"(\b\d{4}[-/.]\d{1,2}[-/.]\d{1,2}\b|\b\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}\b|"
    r"\b(?:\d{1,2}\s+)?(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)(?:\s+\d{1,2}(?:st|nd|rd|th)?)?(?:,?\s+\d{2,4})?\b|"
    r"\b(?:\d{1,2}\s+May|May\s+\d{1,2}(?:st|nd|rd|th)?|May\s+\d{4})\b|"
    r"\b\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM|UTC|GMT|[A-Z]{3})\b)",
    re.IGNORECASE,
)

NUMBER_PATTERN = re.compile(
    r"((?:[\$€£¥]\s*\d+(?:,\d{3})*(?:\.\d+)?)|"
    r"(?:\b\d+(?:\.\d+)?\s*(?:%|px|ms|s|min|MB|GB|TB|KB|kg|g|m|km|dollars?|cents?|USD|EUR)\b)|"
    r"(?:\bv?\d+\.\d+(?:\.\d+)?\b)|"
    r"(?:[<>=]=?\s*\d+(?:\.\d+)?))",
    re.IGNORECASE,
)

ID_PATTERN = re.compile(
    r"(\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b|"
    r"\b0x[0-9a-fA-F]+\b|\b[0-9a-fA-F]{32,64}\b|"
    r"\b(?:id|uuid|key|token|hash|sha|commit|acct|order|ord|ticket|sup|usr|user)[\s:=_-]+['\"]?[a-zA-Z0-9_.-]{3,}['\"]?|"
    r"\b[A-Z]{2,}[-_][A-Z0-9_-]+\b|\b[A-Z]+-[0-9]+\b|"
    r"\b[a-z0-9]+(?:-[a-z0-9]+){2,}\b|\b[a-zA-Z0-9]+_[a-zA-Z0-9_]+\b)",
)

NEGATION_PATTERN = re.compile(
    r"\b(not|no|never|none|neither|nor|without|hardly|scarcely|cannot|can't|won't|don't|doesn't|didn't|shouldn't|mustn't|wouldn't|couldn't|isn't|aren't|wasn't|weren't|hasn't|haven't|hadn't)\b",
    re.IGNORECASE,
)

STRUCTURED_DATA_PATTERN = re.compile(
    r"((?:^|\n)\|[^\n]+\|\n\|[-:\s|]+\|\n\|[^\n]+\||"
    r"(?:\{[\s\S]*?\}|\[[\s\S]*?\])|"
    r"(?:^|\n)\s*[\w.-]+:\s+[^\n]+|"
    r"(?:^|\n)\[[a-zA-Z0-9_.-]+\]|"
    r"(?:^|\n)\s*[-*]\s+\w+:|"
    r"<[a-zA-Z0-9_-]+(\s+[^>]+)?>[\s\S]*?<\/[a-zA-Z0-9_-]+>|"
    r"(?:^|\n)[^\n,]+,[^\n,]+,[^\n,]+(?:\n[^\n,]+,[^\n,]+,[^\n,]+)+)",
    re.MULTILINE,
)


def detect_protected_spans(
    text: str,
    source_type: str,
    is_recent_user_turn: bool = False,
    protect_system_prompt: bool = True,
    protect_code: bool = True,
    protect_numbers: bool = True,
    protect_dates: bool = True,
    protect_ids: bool = True,
    protect_negations: bool = True,
    protect_structured: bool = True,
) -> ProtectionMatch:
    reasons: list[str] = []

    if source_type == "system" and protect_system_prompt:
        reasons.append("system instructions protected by default")

    if is_recent_user_turn:
        reasons.append("recent user turn protected")

    if protect_code and CODE_BLOCK_PATTERN.search(text):
        reasons.append("contains code or syntax block")

    if protect_dates and DATE_PATTERN.search(text):
        reasons.append("contains date/time entity")

    if protect_ids and ID_PATTERN.search(text):
        reasons.append("contains identifier/hash/key")

    if protect_structured and STRUCTURED_DATA_PATTERN.search(text):
        reasons.append("contains structured table/json/yaml")

    if protect_negations and NEGATION_PATTERN.search(text):
        reasons.append("contains critical negation")

    if protect_numbers and NUMBER_PATTERN.search(text):
        reasons.append("contains numeric value/metric")

    is_protected = len(reasons) > 0
    return ProtectionMatch(is_protected=is_protected, reasons=reasons)
