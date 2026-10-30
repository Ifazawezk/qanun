"""Hybrid retrieval: BM25 and dense rankings merged with reciprocal rank fusion.

RRF only looks at ranks, so the two retrievers do not need comparable scores:

    score(d) = sum over retrievers of  weight / (k + rank(d))

k = 60 is the value from the original RRF paper (Cormack et al., 2009) and is
not tuned on the evaluation set.
"""

from __future__ import annotations

from typing import Iterable, Sequence

from qanun.retrieval.base import Hit, Retriever


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
        laws = list(laws) if laws else None
        return {r.name: r.search(query, k=self.depth, laws=laws) for r in self.retrievers}

    def search(self, query: str, k: int = 10, laws: Iterable[str] | None = None) -> list[Hit]:
        rankings = self.rankings(query, laws)
        return reciprocal_rank_fusion(list(rankings.values()), self.k, self.weights)[:k]
