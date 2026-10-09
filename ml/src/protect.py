import re
from typing import NamedTuple


class ProtectionMatch(NamedTuple):
    is_protected: bool
    reasons: list[str]


CODE_BLOCK_PATTERN = re.compile(
    r"(```[\s\S]*?```|`[^`\n]{4,}`|(?:\b(?:def|class|function|const|let|var|import|from|return|public|private|static)\s+\w+)|(?:\{\s*[\"'\w]+\s*:[\s\S]*?\})|(?:SELECT\s+.+\s+FROM\s+\w+))",
    re.IGNORECASE,
)

DATE_PATTERN = re.compile(
    r"(\b\d{4}[-/.]\d{1,2}[-/.]\d{1,2}\b|\b\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}\b|\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2}(?:st|nd|rd|th)?(?:,\s*\d{4})?\b)",
    re.IGNORECASE,
)

NUMBER_PATTERN = re.compile(
    r"(\b(?:\$|€|£|¥)?\d+(?:,\d{3})*(?:\.\d+)?(?:%|px|ms|s|min|h|kg|g|m|km|MB|GB|TB|KB|k|M|B)?\b)",
    re.IGNORECASE,
)

ID_PATTERN = re.compile(
    r"(\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b|\b0x[0-9a-fA-F]+\b|\b(?:id|uuid|key|token|hash|sha|commit)[\s:=_]+['\"]?[a-zA-Z0-9_-]{4,}['\"]?|\b[A-Z0-9_-]{8,}\b)",
    re.IGNORECASE,
)

NEGATION_PATTERN = re.compile(
    r"\b(not|no|never|none|neither|nor|without|hardly|scarcely|cannot|can't|won't|don't|doesn't|didn't|shouldn't|mustn't|wouldn't|couldn't)\b",
    re.IGNORECASE,
)

STRUCTURED_DATA_PATTERN = re.compile(
    r"(\|[^\n]+\|\n\|[-:\s|]+\|\n\|[^\n]+\||^\s*[\w.-]+:\s+[^\n]+$|<[a-zA-Z0-9_-]+(\s+[^>]+)?>[\s\S]*?<\/[a-zA-Z0-9_-]+>|^\s*[-*]\s+\w+:)",
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
