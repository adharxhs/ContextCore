import pytest
from ml.src.scorers.bm25 import BM25Scorer
from ml.src.scorers.dense import DenseScorer
from ml.src.scorers.factory import get_scorer
from ml.src.scorers.hybrid import HybridScorer


def test_bm25_scorer():
    scorer = BM25Scorer()
    texts = [
        "Python is a popular programming language for data science",
        "Cooking recipes for delicious Italian pasta dishes",
        "FastAPI is a modern web framework for Python",
    ]
    scores = scorer.score("python web framework", texts)
    assert len(scores) == 3
    assert scores[2] > scores[1]


def test_dense_scorer():
    scorer = DenseScorer()
    texts = [
        "Antigravity is building autonomous agents",
        "Baking sourdough bread at home",
    ]
    scores = scorer.score("artificial intelligence agents", texts)
    assert len(scores) == 2
    assert scores[0] > scores[1]


def test_hybrid_scorer():
    scorer = HybridScorer(alpha=0.5)
    texts = [
        "Kubernetes container orchestration platform",
        "Gardening tips for spring vegetables",
    ]
    scores = scorer.score("deploy containers on kubernetes", texts)
    assert len(scores) == 2
    assert scores[0] > scores[1]


def test_scorer_factory():
    assert isinstance(get_scorer("bm25"), BM25Scorer)
    assert isinstance(get_scorer("dense"), DenseScorer)
    assert isinstance(get_scorer("hybrid"), HybridScorer)

    with pytest.raises(ValueError):
        get_scorer("invalid_scorer")
