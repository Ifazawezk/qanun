"""Optional second stage: a multilingual cross-encoder over the top candidates.

A cross-encoder reads the question and a passage together, which is slower but
more precise than comparing embeddings. It needs the embeddings extra, so the
import stays inside the class and the desktop build, which leaves out PyTorch,
is not affected. See reranker_model in app_factory.py for when it is on.
"""

from __future__ import annotations

import logging
import os
from typing import Iterable, Mapping, Protocol, Sequence

import numpy as np

from qanun.corpus.models import Article, Law
from qanun.retrieval.base import Hit
from qanun.retrieval.documents import article_passages
from qanun.text.normalize import detect_language

log = logging.getLogger("qanun.rerank")

RERANKER_PRESETS = {"mmarco": "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"}
RERANK_DEPTH = 20
MAX_CHARS = 900


class PairScorer(Protocol):
    name: str

    def score(self, pairs: Sequence[tuple[str, str]]) -> np.ndarray: ...


class CrossEncoderScorer:
    def __init__(self, name: str, batch_size: int = 8, max_length: int = 512):
        from sentence_transformers import CrossEncoder

        self.name = RERANKER_PRESETS.get(name, name)
        self.batch_size = batch_size
        self.model = CrossEncoder(
            self.name, max_length=max_length, cache_folder=os.environ.get("QANUN_MODEL_CACHE")
        )

    def score(self, pairs: Sequence[tuple[str, str]]) -> np.ndarray:
        scores = self.model.predict(
            [list(p) for p in pairs], batch_size=self.batch_size, show_progress_bar=False
        )
        return np.asarray(scores, dtype=np.float64)


class Reranker:
    """Orders hits by the best score among each article's passages."""

    def __init__(
        self,
        scorer: PairScorer,
        articles: Iterable[Article],
        laws: Mapping[str, Law] | None = None,
        depth: int = RERANK_DEPTH,
    ):
        self.scorer = scorer
        self.depth = depth
        laws = laws or {}
        self.passages = {
            a.id: article_passages(a, laws.get(a.law), MAX_CHARS) for a in articles if a.in_force
        }

    def rerank(self, query: str, hits: Sequence[Hit], k: int, translation: str | None = None) -> list[Hit]:
        """Rerank the first `depth` hits. Each passage is compared with the question
        in its own language when a translation exists."""
        head, tail = list(hits[: self.depth]), list(hits[self.depth :])
        if not head:
            return []
        lang = detect_language(query)
        pairs: list[tuple[str, str]] = []
        owners: list[int] = []
        for position, hit in enumerate(head):
            for passage in self.passages.get(hit.article_id, ()):
                text = translation if translation and passage.lang != lang else query
                pairs.append((text, passage.text))
                owners.append(position)
        if not pairs:
            return list(hits[:k])
        best = np.full(len(head), -np.inf)
        np.maximum.at(best, np.array(owners), self.scorer.score(pairs))
        order = sorted(range(len(head)), key=lambda i: -best[i])
        ranked = [Hit(head[i].article_id, float(best[i]), n) for n, i in enumerate(order, start=1)]
        offset = len(ranked)
        ranked += [Hit(h.article_id, h.score, offset + n) for n, h in enumerate(tail, start=1)]
        return ranked[:k]
