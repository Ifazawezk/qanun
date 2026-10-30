"""Loads the corpus and indexes once and answers search requests."""

from __future__ import annotations

import logging
from typing import Iterable

from qanun.corpus.models import Article, Law
from qanun.corpus.store import load_articles, load_laws
from qanun.retrieval.base import Hit, Retriever
from qanun.retrieval.bm25 import BM25Index
from qanun.retrieval.dense import DEFAULT_MODEL, DenseIndex, load_or_build
from qanun.retrieval.hybrid import HybridRetriever

log = logging.getLogger("qanun.search")

METHODS = ("bm25", "dense", "hybrid")


class SearchEngine:
    def __init__(
        self,
        articles: Iterable[Article],
        laws: Iterable[Law],
        dense: DenseIndex | None = None,
    ):
        self.articles = {a.id: a for a in articles}
        self.laws = {law.key: law for law in laws}
        self.bm25 = BM25Index.from_articles(self.articles.values(), self.laws)
        self.dense = dense
        self.hybrid = HybridRetriever([self.bm25, dense]) if dense else None

    @classmethod
    def load(cls, model: str | None = DEFAULT_MODEL) -> "SearchEngine":
        """Load everything from disk. With model=None, or if the embedding
        model cannot be loaded, only BM25 is available."""
        articles = load_articles()
        laws = load_laws()
        dense = None
        if model:
            try:
                dense = load_or_build(model, articles, {law.key: law for law in laws})
            except ImportError as exc:
                log.warning("dense retrieval disabled, missing dependency: %s", exc)
        return cls(articles, laws, dense)

    @property
    def methods(self) -> list[str]:
        return ["bm25", "dense", "hybrid"] if self.dense else ["bm25"]

    @property
    def default_method(self) -> str:
        return "hybrid" if self.dense else "bm25"

    def retriever(self, method: str | None = None) -> Retriever:
        method = method or self.default_method
        if method == "bm25":
            return self.bm25
        if method == "dense" and self.dense:
            return self.dense
        if method == "hybrid" and self.hybrid:
            return self.hybrid
        raise ValueError(f"search method {method!r} is not available; use one of {self.methods}")

    def search(
        self, query: str, k: int = 10, method: str | None = None, laws: Iterable[str] | None = None
    ) -> list[Hit]:
        return self.retriever(method).search(query, k=k, laws=laws)

    def article(self, article_id: str) -> Article | None:
        return self.articles.get(article_id)

    def find(self, law: str, label: str) -> Article | None:
        return self.articles.get(f"{law}:{label}")
