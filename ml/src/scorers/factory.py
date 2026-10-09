from ml.src.scorers.base import BaseScorer
from ml.src.scorers.bm25 import BM25Scorer
from ml.src.scorers.cross_encoder import CrossEncoderScorer
from ml.src.scorers.dense import DenseScorer
from ml.src.scorers.hybrid import HybridScorer
from ml.src.types import ScorerType


def get_scorer(scorer_name: str | ScorerType) -> BaseScorer:
    name_str = scorer_name.value if isinstance(scorer_name, ScorerType) else str(scorer_name).lower()

    if name_str in ("bm25", "bm_25"):
        return BM25Scorer()
    elif name_str == "dense":
        return DenseScorer()
    elif name_str == "hybrid":
        return HybridScorer()
    elif name_str in ("cross_encoder", "cross-encoder", "crossencoder"):
        return CrossEncoderScorer()
    else:
        raise ValueError(
            f"Unsupported scorer: '{scorer_name}'. Supported options are: 'bm25', 'dense', 'hybrid', 'cross_encoder'."
        )
