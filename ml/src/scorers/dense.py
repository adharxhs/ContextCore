import numpy as np

from ml.src.scorers.base import BaseScorer

_EMBED_MODEL = None
_EMBED_MODEL_FAILURE_REASON = None


def _get_embedding_model():
    global _EMBED_MODEL, _EMBED_MODEL_FAILURE_REASON
    if _EMBED_MODEL is None:
        try:
            from fastembed import TextEmbedding

            _EMBED_MODEL = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
        except ImportError as e:
            _EMBED_MODEL = False
            _EMBED_MODEL_FAILURE_REASON = f"fastembed not available: {str(e)}"
        except Exception as e:
            _EMBED_MODEL = False
            _EMBED_MODEL_FAILURE_REASON = f"failed to load embedding model: {str(e)}"
    return _EMBED_MODEL


class DenseScorer(BaseScorer):
    def score(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        if not query.strip():
            return [0.0] * len(texts)

        embed_model = _get_embedding_model()
        if embed_model:
            try:
                # Embed query and texts
                all_texts = [query] + texts
                embeddings = list(embed_model.embed(all_texts))
                query_vec = embeddings[0]
                doc_vecs = np.array(embeddings[1:])

                # Cosine similarity
                q_norm = np.linalg.norm(query_vec)
                d_norms = np.linalg.norm(doc_vecs, axis=1)

                if q_norm > 0:
                    sims = np.dot(doc_vecs, query_vec) / (d_norms * q_norm + 1e-9)
                    # Rescale [-1, 1] to [0, 1]
                    sims = np.clip((sims + 1.0) / 2.0, 0.0, 1.0)
                    return [round(float(s), 4) for s in sims]
            except Exception:
                pass

        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity

            vect = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
            tfidf_mat = vect.fit_transform([query] + texts)
            sims = cosine_similarity(tfidf_mat[0:1], tfidf_mat[1:]).flatten()
            return [round(float(s), 4) for s in sims]
        except Exception:
            return [0.0] * len(texts)
