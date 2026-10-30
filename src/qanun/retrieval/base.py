"""Shared types for the retrievers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Protocol

import numpy as np


@dataclass(frozen=True)
class Hit:
    article_id: str
    score: float
    rank: int  # 1-based


class Retriever(Protocol):
    name: str

    def search(self, query: str, k: int = 10, laws: Iterable[str] | None = None) -> list[Hit]:
        ...


def top_hits(
    doc_ids: list[str],
    scores: np.ndarray,
    k: int,
    allowed: np.ndarray | None = None,
    min_score: float | None = None,
) -> list[Hit]:
    """Turn a score per document into ranked hits."""
    scores = scores.astype(np.float64, copy=True)
    if allowed is not None:
        scores[~allowed] = -np.inf
    if min_score is not None:
        scores[scores <= min_score] = -np.inf
    k = min(k, len(doc_ids))
    if k <= 0:
        return []
    candidates = np.argpartition(-scores, k - 1)[:k]
    order = candidates[np.argsort(-scores[candidates], kind="stable")]
    return [
        Hit(doc_ids[i], float(scores[i]), rank)
        for rank, i in enumerate((i for i in order if np.isfinite(scores[i])), start=1)
    ]


def law_mask(doc_laws: list[str], laws: Iterable[str] | None) -> np.ndarray | None:
    if not laws:
        return None
    wanted = set(laws)
    return np.array([law in wanted for law in doc_laws], dtype=bool)
