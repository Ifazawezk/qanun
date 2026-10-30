"""Question answering on top of the search engine.

Without a language model the assistant is extractive: it returns the best
matching articles with the most relevant passage of each highlighted. With a
model configured it also writes a grounded answer (see generate.py).
"""

from __future__ import annotations

import logging
from typing import Iterable, Sequence

import numpy as np

from qanun.answer.highlight import best_passage
from qanun.answer.models import Answer, Citation
from qanun.corpus.models import Article
from qanun.search import SearchEngine
from qanun.text.normalize import detect_language

log = logging.getLogger("qanun.assistant")

EXTRACTIVE_LEAD = {
    "en": "These are the articles that best match your question. The most relevant "
    "passage of each one is highlighted; read the full article before relying on it.",
    "ar": "هذه هي المواد الأقرب إلى سؤالك، مع تمييز الجزء الأكثر صلة في كل مادة. "
    "يرجى قراءة نص المادة كاملاً قبل الاعتماد عليه.",
}
NO_RESULTS = {
    "en": "No article in the corpus matched this question.",
    "ar": "لم يتم العثور على مادة تطابق هذا السؤال.",
}


class Assistant:
    def __init__(self, engine: SearchEngine, generator=None, top_k: int = 4):
        self.engine = engine
        self.generator = generator
        self.top_k = top_k

    @property
    def can_generate(self) -> bool:
        return self.generator is not None

    # ------------------------------------------------------------ helpers

    def _similarity(self, question: str):
        """Sentence scorer from the embedding model, centred so that only
        differences between sentences count."""
        dense = self.engine.dense
        if dense is None:
            return None
        query = dense.encoder.encode_queries([question])[0]

        def score(sentences: Sequence[str]) -> list[float]:
            sims = dense.encoder.encode_passages(list(sentences)) @ query
            return list(sims - float(np.mean(sims)))

        return score

    def citation(self, article: Article, rank: int, score: float, question: str, lang: str, similarity=None) -> Citation:
        law = self.engine.laws.get(article.law)
        text_lang = "en" if lang == "en" and article.text_en else "ar"
        passage = best_passage(
            question, article.text(text_lang), text_lang, similarity, idf=self.engine.bm25.idf
        )
        return Citation(
            article_id=article.id,
            law=article.law,
            label=article.label,
            title=article.title_en if lang == "en" else article.title_ar,
            law_name=(law.name_en if lang == "en" else law.name_ar) if law else article.law,
            url=article.source_url,
            rank=rank,
            score=score,
            passage=passage,
        )

    # ------------------------------------------------------------ main entry

    def ask(
        self,
        question: str,
        k: int | None = None,
        laws: Iterable[str] | None = None,
        mode: str = "auto",
        method: str | None = None,
    ) -> Answer:
        if mode not in ("auto", "extractive", "generative"):
            raise ValueError(f"unknown mode: {mode}")
        if mode == "generative" and not self.can_generate:
            raise RuntimeError("generative mode needs an LLM; set OPENAI_API_KEY or ANTHROPIC_API_KEY")
        question = question.strip()
        lang = detect_language(question)
        hits = self.engine.search(question, k=k or self.top_k, method=method, laws=laws)
        articles = [self.engine.article(h.article_id) for h in hits]
        similarity = self._similarity(question) if hits else None
        citations = [
            self.citation(article, hit.rank, hit.score, question, lang, similarity)
            for hit, article in zip(hits, articles)
        ]

        if not citations:
            return Answer(question, lang, "extractive", NO_RESULTS[lang], [], refused=True)

        if mode == "extractive" or (mode == "auto" and not self.can_generate):
            return Answer(question, lang, "extractive", EXTRACTIVE_LEAD[lang], citations)

        return self.generator.answer(question, lang, articles, citations)
