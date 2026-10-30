"""Grounded answer generation with an LLM.

The model sees only the retrieved articles and must cite them by id. After
generation every citation is checked against the ids it was given: unknown ids
are removed, and an answer left without a single valid citation is replaced by
a refusal, so the user never gets an uncited claim.
"""

from __future__ import annotations

import logging
import re
from dataclasses import replace
from functools import lru_cache
from importlib import resources
from typing import Sequence

from qanun.answer.assistant import EXTRACTIVE_LEAD
from qanun.answer.llm import LLMClient, LLMError
from qanun.answer.models import Answer, Citation
from qanun.corpus.models import Article, Law

log = logging.getLogger("qanun.generate")

CITATION = re.compile(r"\[([a-z0-9-]+:[a-z0-9-]+)\]")
REFUSAL_TOKEN = "NOT_COVERED"
LANGUAGE_NAMES = {"en": "English", "ar": "Arabic"}
REFUSAL = {
    "en": "The retrieved articles do not answer this question, so I cannot answer it from the law "
    "texts in this corpus. The closest articles are listed below.",
    "ar": "المواد المسترجعة لا تجيب عن هذا السؤال، لذلك لا يمكنني الإجابة عنه من نصوص القوانين "
    "المتوفرة. أقرب المواد مذكورة أدناه.",
}
LLM_FAILED = {
    "en": "The language model could not be reached, so this is the extractive result.",
    "ar": "تعذر الاتصال بالنموذج اللغوي، لذلك هذه نتيجة البحث في النصوص.",
}


@lru_cache(maxsize=None)
def load_prompt(name: str) -> str:
    return resources.files("qanun.prompts").joinpath(f"{name}.md").read_text(encoding="utf-8")


def status_line(article: Article) -> str:
    parts = ["in force" if article.in_force else "repealed"]
    for amendment in article.amendments:
        name = amendment.instrument_en or amendment.instrument_ar
        verb = "added by" if amendment.change == "added" else "amended by"
        parts.append(f"{verb} {name}")
    return ", ".join(parts)


def format_articles(articles: Sequence[Article], laws: dict[str, Law]) -> str:
    template = load_prompt("article")
    blocks = []
    for article in articles:
        law = laws.get(article.law)
        english = (
            f"English (unofficial translation):\n{article.text_en}\n" if article.text_en else ""
        )
        blocks.append(
            template.format_map(
                {
                    "id": article.id,
                    "title": article.title_en,
                    "law": law.name_en if law else article.law,
                    "status": status_line(article),
                    "text_ar": article.text_ar,
                    "english": english,
                }
            ).strip()
        )
    return "\n\n".join(blocks)


def build_messages(
    question: str, lang: str, articles: Sequence[Article], laws: dict[str, Law]
) -> tuple[str, str]:
    language = LANGUAGE_NAMES[lang]
    system = load_prompt("system").format_map({"language": language})
    user = load_prompt("user").format_map(
        {"articles": format_articles(articles, laws), "question": question, "language": language}
    )
    return system, user


def check_citations(text: str, allowed: set[str]) -> tuple[str, list[str], list[str]]:
    """Return the text without invalid citations, the valid ids in order of
    first use, and the invalid ids that were removed."""
    valid: list[str] = []
    invalid: list[str] = []

    def keep(match: re.Match) -> str:
        article_id = match.group(1)
        if article_id in allowed:
            if article_id not in valid:
                valid.append(article_id)
            return match.group(0)
        if article_id not in invalid:
            invalid.append(article_id)
        return ""

    cleaned = CITATION.sub(keep, text)
    cleaned = re.sub(r"[ \t]+([.,;:])", r"\1", cleaned)
    return cleaned.strip(), valid, invalid


class Generator:
    def __init__(self, client: LLMClient, laws: dict[str, Law]):
        self.client = client
        self.laws = laws

    def answer(
        self,
        question: str,
        lang: str,
        articles: Sequence[Article],
        citations: Sequence[Citation],
    ) -> Answer:
        model = f"{self.client.provider}:{self.client.model}"
        system, user = build_messages(question, lang, articles, self.laws)
        try:
            raw = self.client.complete(system, user)
        except LLMError as exc:
            log.warning("falling back to extractive answer: %s", exc)
            return Answer(
                question, lang, "extractive", EXTRACTIVE_LEAD[lang], list(citations),
                notes=[LLM_FAILED[lang]],
            )

        if raw.strip().upper().startswith(REFUSAL_TOKEN):
            return Answer(question, lang, "generative", REFUSAL[lang], list(citations), refused=True, model=model)

        text, valid, invalid = check_citations(raw, {a.id for a in articles})
        notes = [f"Removed citation to {i}, which was not among the retrieved articles." for i in invalid]
        if not valid:
            notes.append("The model's answer cited no retrieved article, so it was withheld.")
            return Answer(
                question, lang, "generative", REFUSAL[lang], list(citations),
                refused=True, model=model, notes=notes,
            )

        order = {article_id: i for i, article_id in enumerate(valid)}
        ordered = sorted(
            (replace(c, cited=c.article_id in order) for c in citations),
            key=lambda c: (order.get(c.article_id, len(order)), c.rank),
        )
        return Answer(question, lang, "generative", text, ordered, model=model, notes=notes)
