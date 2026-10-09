import math
from ml.src.scorers.base import BaseScorer
from ml.src.scorers.hybrid import HybridScorer

_CROSS_ENCODER_MODEL = None
_CROSS_ENCODER_FAILURE_REASON = None


def _get_cross_encoder():
    global _CROSS_ENCODER_MODEL, _CROSS_ENCODER_FAILURE_REASON
    if _CROSS_ENCODER_MODEL is None:
        try:
            from fastembed.rerank.cross_encoder import TextCrossEncoder

            _CROSS_ENCODER_MODEL = TextCrossEncoder(model_name="BAAI/bge-reranker-base")
        except ImportError as e:
            _CROSS_ENCODER_MODEL = False
            _CROSS_ENCODER_FAILURE_REASON = f"fastembed reranker not available: {str(e)}"
        except Exception as e:
            _CROSS_ENCODER_MODEL = False
            _CROSS_ENCODER_FAILURE_REASON = f"failed to load cross-encoder model: {str(e)}"
    return _CROSS_ENCODER_MODEL


class CrossEncoderScorer(BaseScorer):
    def __init__(self):
        self._hybrid_fallback = HybridScorer()

    def score(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        if not query.strip():
            return [0.0] * len(texts)

        model = _get_cross_encoder()
        if model:
            try:
                raw_scores = list(model.rerank(query, texts))
                # Apply sigmoid normalization: 1 / (1 + exp(-s))
                norm_scores = []
                for s in raw_scores:
                    s_val = float(s)
                    prob = 1.0 / (1.0 + math.exp(-s_val))
                    norm_scores.append(round(prob, 4))
                return norm_scores
            except Exception:
                pass

        return self._hybrid_fallback.score(query, texts)
