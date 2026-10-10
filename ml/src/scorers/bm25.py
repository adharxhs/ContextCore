import math
import re

import numpy as np

from ml.src.scorers.base import BaseScorer


def _tokenize(text: str) -> list[str]:
    return [w for w in re.findall(r"\w+", text.lower()) if len(w) > 0]


class BM25Scorer(BaseScorer):
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b

    def score(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        if not query.strip():
            return [0.0] * len(texts)

        tokenized_corpus = [_tokenize(doc) for doc in texts]
        tokenized_query = _tokenize(query)
        if not tokenized_query:
            return [0.0] * len(texts)

        corpus_size = len(tokenized_corpus)
        doc_len = [len(doc) for doc in tokenized_corpus]
        avgdl = sum(doc_len) / corpus_size if corpus_size > 0 else 1.0

        # Term frequencies per document and document frequencies
        doc_freqs: list[dict[str, int]] = []
        nd: dict[str, int] = {}
        for doc in tokenized_corpus:
            freqs: dict[str, int] = {}
            for word in doc:
                freqs[word] = freqs.get(word, 0) + 1
            doc_freqs.append(freqs)
            for word in freqs:
                nd[word] = nd.get(word, 0) + 1

        # Lucene-style positive IDF
        idf: dict[str, float] = {}
        for word, freq in nd.items():
            idf[word] = math.log(1.0 + (corpus_size - freq + 0.5) / (freq + 0.5))

        scores = np.zeros(corpus_size)
        doc_len_arr = np.array(doc_len)

        for q in tokenized_query:
            if q not in idf:
                continue
            q_idf = idf[q]
            q_freq = np.array([df.get(q, 0) for df in doc_freqs])
            denom = q_freq + self.k1 * (1.0 - self.b + self.b * doc_len_arr / (avgdl or 1.0))
            # Avoid division by zero
            denom = np.where(denom == 0, 1.0, denom)
            scores += q_idf * (q_freq * (self.k1 + 1.0) / denom)

        max_score = float(np.max(scores)) if len(scores) > 0 else 0.0
        if max_score <= 0.0:
            return [0.0] * len(texts)

        return [round(float(s) / max_score, 4) for s in scores]
