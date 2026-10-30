"""What the assistant returns."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Passage:
    """A sentence or two from an article, with character ranges to highlight."""

    text: str
    lang: str
    highlights: list[tuple[int, int]] = field(default_factory=list)


@dataclass
class Citation:
    article_id: str
    law: str
    label: str
    title: str
    law_name: str
    url: str
    rank: int
    score: float
    passage: Passage | None = None


@dataclass
class Answer:
    question: str
    language: str  # language of the question, "ar" or "en"
    mode: str  # "generative" or "extractive"
    text: str  # the answer (generative) or a short lead-in (extractive)
    citations: list[Citation]
    refused: bool = False
    model: str | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
