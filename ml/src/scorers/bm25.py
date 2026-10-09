import re
from rank_bm25 import BM25Okapi

from ml.src.scorers.base import BaseScorer


def _tokenize(text: str) -> list[str]:
    return [w for w in re.findall(r"\w+", text.lower()) if len(w) > 1]


class BM25Scorer(BaseScorer):
    def score(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        if not query.strip():
            return [0.0] * len(texts)

        tokenized_corpus = [_tokenize(doc) for doc in texts]
        # Avoid empty tokenized docs failing BM25
        tokenized_corpus = [doc if doc else ["_empty_"] for doc in tokenized_corpus]

        bm25 = BM25Okapi(tokenized_corpus)
        tokenized_query = _tokenize(query)
        if not tokenized_query:
            return [0.0] * len(texts)

        raw_scores = bm25.get_scores(tokenized_query)
        max_score = float(max(raw_scores)) if len(raw_scores) > 0 else 0.0

        if max_score <= 0.0:
            return [0.0] * len(texts)

        # Normalize relative to max score
        return [round(float(s) / max_score, 4) for s in raw_scores]
