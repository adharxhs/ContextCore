from abc import ABC, abstractmethod


class BaseScorer(ABC):
    @abstractmethod
    def score(self, query: str, texts: list[str]) -> list[float]:
        """Returns relative relevance score for each text given the query.

        Scores should be normalized floats >= 0.0.
        """
        pass
