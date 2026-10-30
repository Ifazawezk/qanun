"""Request and response models for the HTTP API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class PassageOut(BaseModel):
    text: str
    lang: str
    highlights: list[tuple[int, int]]


class CitationOut(BaseModel):
    article_id: str
    law: str
    label: str
    title: str
    law_name: str
    url: str
    rank: int
    score: float
    cited: bool
    passage: PassageOut | None


class AskRequest(BaseModel):
    question: str = Field(min_length=2, max_length=1000)
    k: int = Field(default=4, ge=1, le=10)
    mode: Literal["auto", "extractive", "generative"] = "auto"
    laws: list[str] | None = None


class AnswerOut(BaseModel):
    question: str
    language: str
    mode: str
    text: str
    refused: bool
    model: str | None
    notes: list[str]
    citations: list[CitationOut]


class SearchHit(BaseModel):
    rank: int
    score: float
    article_id: str
    law: str
    label: str
    title_en: str
    title_ar: str
    passage: PassageOut | None


class SearchOut(BaseModel):
    query: str
    method: str
    hits: list[SearchHit]


class AmendmentOut(BaseModel):
    instrument_ar: str
    instrument_en: str | None
    instrument_article: str | None
    effective_date: str | None
    change: str


class PreviousVersionOut(BaseModel):
    text_ar: str
    effective_from: str | None
    effective_to: str | None


class ArticleOut(BaseModel):
    id: str
    law: str
    law_name_en: str
    law_name_ar: str
    label: str
    number: int
    kind: str
    title_en: str
    title_ar: str
    chapter_en: str | None
    chapter_ar: str | None
    section_en: str | None
    section_ar: str | None
    text_ar: str
    text_en: str | None
    text_en_source: str | None
    status: str
    effective_date: str | None
    source_url: str
    amendments: list[AmendmentOut]
    history: list[PreviousVersionOut]
    notes: list[str]


class ArticleSummary(BaseModel):
    id: str
    law: str
    label: str
    title_en: str
    title_ar: str
    section_en: str | None
    section_ar: str | None
    status: str
    has_english: bool


class LawOut(BaseModel):
    key: str
    name_en: str
    name_ar: str
    title_ar: str
    title_en: str | None
    number: int
    year: int
    status: str
    url: str
    article_count: int


class HealthOut(BaseModel):
    status: str
    articles: int
    search_methods: list[str]
    embedding_model: str | None
    generation: bool
    llm: str | None
