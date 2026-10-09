from ml.src.scorers.base import BaseScorer
from ml.src.scorers.bm25 import BM25Scorer
from ml.src.scorers.dense import DenseScorer


class HybridScorer(BaseScorer):
    def __init__(self, alpha: float = 0.5):
        self.alpha = alpha
        self.bm25 = BM25Scorer()
        self.dense = DenseScorer()

    def score(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        if not query.strip():
            return [0.0] * len(texts)

        bm25_scores = self.bm25.score(query, texts)
        dense_scores = self.dense.score(query, texts)

        hybrid_scores = []
        for b, d in zip(bm25_scores, dense_scores, strict=True):
            score = self.alpha * b + (1.0 - self.alpha) * d
            hybrid_scores.append(round(score, 4))

        return hybrid_scores
