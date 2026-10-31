"""Loads the corpus and indexes once and answers search requests."""

from __future__ import annotations

import logging
from typing import Iterable

from qanun.corpus.models import Article, Law
from qanun.corpus.store import load_articles, load_laws
from qanun.retrieval.base import Hit, Retriever
from qanun.retrieval.bm25 import BM25Index
from qanun.retrieval.dense import DEFAULT_MODEL, DenseIndex, load_or_build
from qanun.retrieval.hybrid import CrossLanguageRetriever, HybridRetriever
from qanun.retrieval.rerank import RERANK_DEPTH, CrossEncoderScorer, Reranker

log = logging.getLogger("qanun.search")

METHODS = ("bm25", "dense", "hybrid", "cross")


class SearchEngine:
    def __init__(
        self,
        articles: Iterable[Article],
        laws: Iterable[Law],
        dense: DenseIndex | None = None,
        reranker: Reranker | None = None,
    ):
        self.articles = {a.id: a for a in articles}
        self.laws = {law.key: law for law in laws}
        self.bm25 = BM25Index.from_articles(self.articles.values(), self.laws)
        self.dense = dense
        self.hybrid = HybridRetriever([self.bm25, dense]) if dense else None
        self.cross = CrossLanguageRetriever(self.bm25, dense)
        self.reranker = reranker

    @classmethod
    def load(cls, model: str | None = DEFAULT_MODEL, reranker: str | None = None) -> "SearchEngine":
        """Load everything from disk. With model=None, or if the embedding
        model cannot be loaded, only BM25 is available."""
        articles = load_articles()
        laws = load_laws()
        by_key = {law.key: law for law in laws}
        dense = None
        if model:
            try:
                dense = load_or_build(model, articles, by_key)
            except ImportError as exc:
                log.warning("dense retrieval disabled, missing dependency: %s", exc)
        ranker = None
        if reranker:
            try:
                ranker = Reranker(CrossEncoderScorer(reranker), articles, by_key)
            except Exception as exc:
                # It is optional and may be missing from the cache or unable to download,
                # and neither should stop search from starting.
                log.warning("reranker disabled: %s", type(exc).__name__)
        return cls(articles, laws, dense, ranker)

    @property
    def methods(self) -> list[str]:
        return ["bm25", "dense", "hybrid", "cross"] if self.dense else ["bm25"]

    @property
    def default_method(self) -> str:
        return "cross" if self.dense else "bm25"

    def retriever(self, method: str | None = None) -> Retriever:
        method = method or self.default_method
        if method == "bm25":
            return self.bm25
        if method == "dense" and self.dense:
            return self.dense
        if method == "hybrid" and self.hybrid:
            return self.hybrid
        if method == "cross" and self.dense:
            return self.cross
        raise ValueError(f"search method {method!r} is not available; use one of {self.methods}")

    def search(
        self,
        query: str,
        k: int = 10,
        method: str | None = None,
        laws: Iterable[str] | None = None,
        translation: str | None = None,
        rerank: bool = True,
    ) -> list[Hit]:
        """A translation applies to the default search, and also works with BM25
        alone, which is how the desktop build reaches Arabic-only articles. The
        reranker applies to the default search only, so a named method stays pure."""
        ranker = self.reranker if rerank and method is None else None
        depth = max(k, RERANK_DEPTH) if ranker else k
        if translation and method in (None, "cross"):
            hits = self.cross.search(query, k=depth, laws=laws, translation=translation)
        else:
            hits = self.retriever(method).search(query, k=depth, laws=laws)
        if ranker:
            return ranker.rerank(query, hits, k, translation)
        return hits[:k]

    def article(self, article_id: str) -> Article | None:
        return self.articles.get(article_id)

    def find(self, law: str, label: str) -> Article | None:
        return self.articles.get(f"{law}:{label}")
