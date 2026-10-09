import pytest
from ml.src.scorers.bm25 import BM25Scorer
from ml.src.scorers.cross_encoder import CrossEncoderScorer
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
    assert scores[0] > scores[1]
    assert scores[2] == 1.0

    # Test empty inputs
    assert scorer.score("", texts) == [0.0, 0.0, 0.0]
    assert scorer.score("python", []) == []

    # Test single and two-document collections (Lucene non-zero IDF)
    single_score = scorer.score("python", ["python"])
    assert single_score == [1.0]
    two_scores = scorer.score("what is python", ["python is a programming language", "pasta recipe with cheese"])
    assert two_scores[0] > two_scores[1]


def test_dense_scorer():
    scorer = DenseScorer()
    texts = [
        "Antigravity is building autonomous agents",
        "Baking sourdough bread at home",
    ]
    scores = scorer.score("artificial intelligence agents", texts)
    assert len(scores) == 2
    assert scores[0] > scores[1]

    # Test empty inputs
    assert scorer.score("", texts) == [0.0, 0.0]
    assert scorer.score("test", []) == []


def test_hybrid_scorer():
    scorer = HybridScorer(alpha=0.5)
    texts = [
        "Kubernetes container orchestration platform",
        "Gardening tips for spring vegetables",
    ]
    scores = scorer.score("deploy containers on kubernetes", texts)
    assert len(scores) == 2
    assert scores[0] > scores[1]

    # Test empty inputs
    assert scorer.score("", texts) == [0.0, 0.0]
    assert scorer.score("test", []) == []


def test_cross_encoder_scorer():
    scorer = CrossEncoderScorer()
    texts = [
        "Rust provides memory safety without garbage collection",
        "Sourdough baking guide and flour hydration",
    ]
    scores = scorer.score("memory safety in rust", texts)
    assert len(scores) == 2
    assert scores[0] > scores[1]

    # Test empty inputs
    assert scorer.score("", texts) == [0.0, 0.0]
    assert scorer.score("test", []) == []


def test_scorer_factory():
    assert isinstance(get_scorer("bm25"), BM25Scorer)
    assert isinstance(get_scorer("dense"), DenseScorer)
    assert isinstance(get_scorer("hybrid"), HybridScorer)
    assert isinstance(get_scorer("cross_encoder"), CrossEncoderScorer)

    with pytest.raises(ValueError):
        get_scorer("invalid_scorer")
