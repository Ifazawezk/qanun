"""Okapi BM25 over normalized, lightly stemmed Arabic and English terms."""

from __future__ import annotations

import math
from collections import Counter
from typing import Iterable, Sequence

import numpy as np

from qanun.corpus.models import Article, Law
from qanun.retrieval.base import Hit, law_mask, top_hits
from qanun.retrieval.documents import article_document
from qanun.text.tokenize import analyze


class BM25Index:
    name = "bm25"

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_ids: list[str] = []
        self.doc_laws: list[str] = []
        self.doc_len = np.zeros(0)
        self.avgdl = 0.0
        self.postings: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self.idf: dict[str, float] = {}

    def fit(
        self, doc_ids: Sequence[str], texts: Sequence[str], doc_laws: Sequence[str] | None = None
    ) -> "BM25Index":
        self.doc_ids = list(doc_ids)
        self.doc_laws = list(doc_laws) if doc_laws else [""] * len(self.doc_ids)
        lists: dict[str, list[tuple[int, int]]] = {}
        lengths = []
        for idx, text in enumerate(texts):
            terms = analyze(text)
            lengths.append(len(terms))
            for term, tf in Counter(terms).items():
                lists.setdefault(term, []).append((idx, tf))
        self.doc_len = np.array(lengths, dtype=np.float64)
        self.avgdl = float(self.doc_len.mean()) if len(lengths) else 0.0
        n = len(self.doc_ids)
        self.postings = {}
        self.idf = {}
        for term, entries in lists.items():
            docs = np.array([d for d, _ in entries], dtype=np.int64)
            tfs = np.array([t for _, t in entries], dtype=np.float64)
            self.postings[term] = (docs, tfs)
            df = len(entries)
            self.idf[term] = math.log(1 + (n - df + 0.5) / (df + 0.5))
        return self

    @classmethod
    def from_articles(
        cls, articles: Iterable[Article], laws: dict[str, Law] | None = None, **kwargs
    ) -> "BM25Index":
        docs = [a for a in articles if a.in_force]
        laws = laws or {}
        return cls(**kwargs).fit(
            [a.id for a in docs],
            [article_document(a, laws.get(a.law)) for a in docs],
            [a.law for a in docs],
        )

    def scores(self, query: str) -> np.ndarray:
        scores = np.zeros(len(self.doc_ids))
        if not self.doc_ids:
            return scores
        norm = self.k1 * (1 - self.b + self.b * self.doc_len / max(self.avgdl, 1e-9))
        for term in dict.fromkeys(analyze(query)):
            posting = self.postings.get(term)
            if posting is None:
                continue
            docs, tfs = posting
            scores[docs] += self.idf[term] * tfs * (self.k1 + 1) / (tfs + norm[docs])
        return scores

    def search(self, query: str, k: int = 10, laws: Iterable[str] | None = None) -> list[Hit]:
        return top_hits(
            self.doc_ids, self.scores(query), k, law_mask(self.doc_laws, laws), min_score=0.0
        )
