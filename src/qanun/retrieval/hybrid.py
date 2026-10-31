"""Hybrid retrieval: BM25 and dense rankings merged with reciprocal rank fusion.

RRF only looks at ranks, so the two retrievers do not need comparable scores:

    score(d) = sum over retrievers of  weight / (k + rank(d))

k = 60 is the value from the original RRF paper (Cormack et al., 2009) and is
not tuned on the evaluation set.
"""

from __future__ import annotations

from typing import Iterable, Sequence

from qanun.retrieval.base import Hit, Retriever
from qanun.text.normalize import detect_language


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[Hit]],
    k: int = 60,
    weights: Sequence[float] | None = None,
) -> list[Hit]:
    weights = weights or [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError("need one weight per ranking")
    fused: dict[str, float] = {}
    for ranking, weight in zip(rankings, weights):
        for hit in ranking:
            fused[hit.article_id] = fused.get(hit.article_id, 0.0) + weight / (k + hit.rank)
    # Ties are broken by first appearance, which favours the earlier ranking.
    order = sorted(fused.items(), key=lambda item: -item[1])
    return [Hit(doc, score, rank) for rank, (doc, score) in enumerate(order, start=1)]


class HybridRetriever:
    name = "hybrid"

    def __init__(
        self,
        retrievers: Sequence[Retriever],
        k: int = 60,
        depth: int = 50,
        weights: Sequence[float] | None = None,
    ):
        self.retrievers = list(retrievers)
        self.k = k
        self.depth = depth
        self.weights = weights

    def rankings(self, query: str, laws: Iterable[str] | None = None) -> dict[str, list[Hit]]:
        """One ranking per retriever, keyed by name (made unique if two share one)."""
        laws = list(laws) if laws else None
        out: dict[str, list[Hit]] = {}
        for retriever in self.retrievers:
            key, n = retriever.name, 2
            while key in out:
                key, n = f"{retriever.name}#{n}", n + 1
            out[key] = retriever.search(query, k=self.depth, laws=laws)
        return out

    def search(self, query: str, k: int = 10, laws: Iterable[str] | None = None) -> list[Hit]:
        rankings = self.rankings(query, laws)
        return reciprocal_rank_fusion(list(rankings.values()), self.k, self.weights)[:k]


def max_fusion(rankings: Sequence[Sequence[Hit]]) -> list[Hit]:
    """CombMAX: an article keeps its best score from any ranking.

    Summing would count an article with text in both languages twice and bury
    the ones that exist only in Arabic, which are the ones this is meant to find.
    """
    best: dict[str, float] = {}
    for ranking in rankings:
        for hit in ranking:
            best[hit.article_id] = max(best.get(hit.article_id, 0.0), hit.score)
    order = sorted(best.items(), key=lambda item: -item[1])
    return [Hit(doc, score, rank) for rank, (doc, score) in enumerate(order, start=1)]


class CrossLanguageRetriever:
    """Searches each language on its own terms, then keeps each article's best result.

    With a translation, the question searches the passages of its own language
    and the translation searches the passages of the other. Without one, the
    question searches both sets separately, so an article that exists only in
    Arabic competes with other Arabic passages and is not always outscored by
    English passages, whose similarity to an English question is higher.
    """

    name = "cross"

    def __init__(self, bm25: Retriever, dense=None, k: int = 60, depth: int = 50):
        self.bm25 = bm25
        self.dense = dense
        self.k = k
        self.depth = depth

    def _hybrid(self, query: str, lang: str, laws: list[str] | None) -> list[Hit]:
        rankings = [self.bm25.search(query, k=self.depth, laws=laws)]
        if self.dense is not None:
            rankings.append(self.dense.search(query, k=self.depth, laws=laws, lang=lang))
        return reciprocal_rank_fusion(rankings, self.k)

    def search(
        self,
        query: str,
        k: int = 10,
        laws: Iterable[str] | None = None,
        translation: str | None = None,
    ) -> list[Hit]:
        laws = list(laws) if laws else None
        lang = detect_language(query)
        other = "ar" if lang == "en" else "en"
        if translation:
            return max_fusion(
                [self._hybrid(query, lang, laws), self._hybrid(translation, other, laws)]
            )[:k]
        if self.dense is None:
            return reciprocal_rank_fusion([self.bm25.search(query, k=self.depth, laws=laws)], self.k)[:k]
        same, across = (
            reciprocal_rank_fusion([self.dense.search(query, k=self.depth, laws=laws, lang=code)], self.k)
            for code in (lang, other)
        )
        keyword = self.bm25.search(query, k=self.depth, laws=laws)
        return reciprocal_rank_fusion([keyword, max_fusion([same, across])], self.k)[:k]
