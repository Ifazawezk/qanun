"""Grounded answer generation with an LLM.

The model sees only the retrieved articles and must cite them by id. Nothing it
writes is trusted. After generation:

- every citation, in whatever form the model wrote it, is checked against the
  articles it was given, including references in prose such as "Article 54";
- a sentence that rests on a citation that cannot be verified is removed;
- an answer left without a valid citation is replaced by a refusal;
- an answer in the wrong language is withheld.

Retrieved text and the question are data. They are escaped so they cannot close
the tags around them, and the system prompt says not to follow them.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field, replace
from functools import lru_cache
from importlib import resources
from typing import Sequence

from qanun.answer.assistant import EXTRACTIVE_LEAD
from qanun.answer.llm import LLMClient, LLMError
from qanun.answer.models import Answer, Citation
from qanun.corpus.models import Article, Law
from qanun.text.normalize import DIGITS, detect_language

log = logging.getLogger("qanun.generate")

CITATION = re.compile(r"\[([a-z0-9-]+:[a-z0-9-]+)\]")
# Upper case on purpose: "not covered by this law" is an ordinary answer.
REFUSAL_TOKEN = re.compile(r"NOT[_ ]COVERED")
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
WRONG_LANGUAGE = {
    "en": "The language model answered in the wrong language, so this is the extractive result.",
    "ar": "أجاب النموذج اللغوي بلغة غير لغة السؤال، لذلك هذه نتيجة البحث في النصوص.",
}
REMOVED = {
    "en": "Removed a reference to {items}, which was not among the retrieved articles.",
    "ar": "حُذفت الإشارة إلى {items} لأنها ليست ضمن المواد المسترجعة.",
}
DROPPED = {
    "en": "Removed {count} sentence(s) that relied on an article that could not be verified.",
    "ar": "حُذفت جمل عددها {count} اعتمدت على مادة تعذر التحقق منها.",
}
WITHHELD = {
    "en": "The model's answer cited no retrieved article, so it was withheld.",
    "ar": "لم تستشهد إجابة النموذج بأي مادة مسترجعة، لذلك تم حجبها.",
}
UNCITED = {
    "en": "Part of this answer carries no citation. Check it against the articles below.",
    "ar": "جزء من هذه الإجابة بلا استشهاد. يرجى مراجعته مع المواد أدناه.",
}

# ---------------------------------------------------------------- finding references

DIGIT = "0-9٠-٩۰-۹"
NUMBER = f"[{DIGIT}]+"
BRACKETED = re.compile(r"[\[【]([^\]】\n]{1,300})[\]】]")
ID_TEXT = re.compile(rf"[A-Za-z][A-Za-z0-9-]*\s*:\s*[A-Za-z{DIGIT}-]+")
BARE_ID = re.compile(rf"(?<![\w\[:-])[A-Za-z][A-Za-z-]*-\d{{4}}:[A-Za-z{DIGIT}-]+(?![\w\]])")

BIS_EN = rf"(?:[\s-]*bis(?:[\s-]*{NUMBER})?)?"
LABEL_EN = rf"{NUMBER}{BIS_EN}"
SEPARATOR_EN = r"\s*(?:,|;|&|\band\b|\bor\b)\s*"
# Plural forms take a list; the singular must not swallow "and 3 weeks".
PROSE_EN = re.compile(
    rf"\b(?:Articles\s+\(?(?P<many>{LABEL_EN}(?:{SEPARATOR_EN}{LABEL_EN})*)"
    rf"|(?:Article|Art\.?)\s+\(?(?P<one>{LABEL_EN}))",
    re.IGNORECASE,
)
BIS_AR = rf"(?:\s*\(?مكرر(?:ا|اً|ًا)?\)?(?:\s*\(?{NUMBER}\)?)?)?"
LABEL_AR = rf"\(?{NUMBER}\)?{BIS_AR}"
SEPARATOR_AR = r"\s*(?:،|,|؛|و|أو)\s*"
PROSE_AR = re.compile(
    rf"(?:(?:المواد|المادتين|المادتان)\s*(?:رقم\s*)?(?P<many>{LABEL_AR}(?:{SEPARATOR_AR}{LABEL_AR})*)"
    rf"|المادة\s*(?:رقم\s*)?(?P<one>{LABEL_AR}))"
)
# Numbers written as words cannot be matched to a label, so they fail closed.
WORD_EN = (
    r"one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|"
    r"sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|"
    r"hundred|first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth"
)
WORD_NUMBER_EN = re.compile(rf"\bArticle\s+(?:{WORD_EN})\b", re.IGNORECASE)
WORD_NUMBER_AR = re.compile(
    "المادة\\s+(?:ال)?(?:أولى|ثانية|ثالثة|رابعة|خامسة|سادسة|سابعة|ثامنة|تاسعة|عاشرة|حادية|مائة)"
)
ANY_LABEL = re.compile(rf"{NUMBER}(?:\s*(?:[\s-]*bis|\(?مكرر(?:ا|اً|ًا)?\)?)(?:[\s()-]*{NUMBER})?)?", re.IGNORECASE)
SENTENCE_END = re.compile(r"(?<=[.!?؟])(?<!\bNo\.)(?<!\bArt\.)\s+")
CITES_AFTER_STOP = re.compile(r"[ \t]*([.!?؟])[ \t]*((?:\[[a-z0-9-]+:[a-z0-9-]+\][ \t]*)+)")


def canonical_id(text: str) -> str:
    return re.sub(r"\s+", "", text.translate(DIGITS)).lower()


def normalize_citations(text: str) -> str:
    """Rewrite every form a model uses for an article id as [law:label], and move a
    citation written after the full stop back inside the sentence it belongs to."""

    def bracket(match: re.Match) -> str:
        ids = ID_TEXT.findall(match.group(1))
        return "".join(f"[{canonical_id(i)}]" for i in ids) if ids else match.group(0)

    text = BARE_ID.sub(lambda m: f"[{canonical_id(m.group(0))}]", BRACKETED.sub(bracket, text))
    return CITES_AFTER_STOP.sub(lambda m: f" {m.group(2).strip()}{m.group(1)} ", text).strip()


def label_key(raw: str) -> str:
    """'52 bis', '52-bis', '(٥٢) مكرر' -> '52-bis'; '115 مكرراً (1)' -> '115-bis-1'."""
    text = raw.translate(DIGITS).lower()
    text = re.sub(r"مكرر(?:ا|اً|ًا)?", " bis ", text)
    return "-".join(re.findall(r"[0-9]+|bis", text))


def prose_labels(sentence: str) -> tuple[list[str], bool]:
    """Article labels mentioned outside brackets, and whether a number was written
    as a word, which cannot be checked."""
    labels: list[str] = []
    for pattern in (PROSE_EN, PROSE_AR):
        for match in pattern.finditer(sentence):
            labels += [label_key(found) for found in ANY_LABEL.findall(match.group("many") or match.group("one"))]
    worded = bool(WORD_NUMBER_EN.search(sentence) or WORD_NUMBER_AR.search(sentence))
    return labels, worded


def known_labels(articles: Sequence[Article]) -> set[str]:
    """Labels that prose may mention. Issuance articles are written as a bare number."""
    labels = {a.label for a in articles}
    return labels | {label.removeprefix("issuance-") for label in labels}


@dataclass
class CheckedAnswer:
    text: str
    valid: list[str] = field(default_factory=list)
    invalid: list[str] = field(default_factory=list)
    dropped: int = 0
    uncited: int = 0


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
    cleaned = re.sub(r"[ \t]+([.,;:،؛])", r"\1", cleaned)
    return cleaned.strip(), valid, invalid


def check_answer(raw: str, articles: Sequence[Article]) -> CheckedAnswer:
    """Validate a reply sentence by sentence, failing closed."""
    allowed = {a.id for a in articles}
    labels = known_labels(articles)
    result = CheckedAnswer("")
    lines = []
    for line in normalize_citations(raw).split("\n"):
        kept = []
        for sentence in SENTENCE_END.split(line):
            cleaned, valid, invalid = check_citations(sentence, allowed)
            mentions, worded = prose_labels(CITATION.sub("", sentence))
            unknown = [label for label in mentions if label not in labels]
            result.invalid += [i for i in [*invalid, *unknown] if i not in result.invalid]
            if unknown or worded or (invalid and not valid):
                # Its evidence is gone, so the claim goes with it.
                result.dropped += sentence.strip() != ""
                continue
            result.valid += [i for i in valid if i not in result.valid]
            if cleaned:
                kept.append(cleaned)
                is_claim = len(cleaned.split()) > 3 and not cleaned.endswith(":")
                result.uncited += is_claim and not valid
        if kept:
            lines.append(" ".join(kept))
    result.text = "\n".join(lines).strip()
    return result


# ---------------------------------------------------------------- building the prompt


@lru_cache(maxsize=None)
def load_prompt(name: str) -> str:
    return resources.files("qanun.prompts").joinpath(f"{name}.md").read_text(encoding="utf-8")


def as_data(text: str) -> str:
    """Stop retrieved text or a question from closing the tags that mark it as data."""
    return text.replace("<", "‹").replace(">", "›").replace('"', "'")


def status_line(article: Article) -> str:
    parts = ["in force" if article.in_force else "repealed"]
    for amendment in article.amendments:
        name = amendment.instrument_en or amendment.instrument_ar
        verb = "added by" if amendment.change == "added" else "amended by"
        parts.append(f"{verb} {name}")
    return ", ".join(parts)


def shorten(text: str, limit: int | None) -> str:
    """Cut to a limit, saying so, for models with a small context window."""
    if limit is None or len(text) <= limit:
        return text
    return text[:limit].rstrip() + " [cut for length]"


def format_articles(
    articles: Sequence[Article], laws: dict[str, Law], max_article_chars: int | None = None
) -> str:
    template = load_prompt("article")
    blocks = []
    for article in articles:
        law = laws.get(article.law)
        text_ar = shorten(article.text_ar, max_article_chars)
        # The official text comes first, so a limit costs the translation before it.
        room = None if max_article_chars is None else max(max_article_chars - len(text_ar), 0)
        english = ""
        if article.text_en and room != 0:
            english = f"English (unofficial translation):\n{as_data(shorten(article.text_en, room))}\n"
        blocks.append(
            template.format_map(
                {
                    "id": article.id,
                    "title": as_data(article.title_en),
                    "law": as_data(law.name_en if law else article.law),
                    "status": as_data(status_line(article)),
                    "text_ar": as_data(text_ar),
                    "english": english,
                }
            ).strip()
        )
    return "\n\n".join(blocks)


def build_messages(
    question: str,
    lang: str,
    articles: Sequence[Article],
    laws: dict[str, Law],
    max_article_chars: int | None = None,
) -> tuple[str, str]:
    language = LANGUAGE_NAMES[lang]
    system = load_prompt("system").format_map({"language": language})
    user = load_prompt("user").format_map(
        {
            "articles": format_articles(articles, laws, max_article_chars),
            "question": as_data(question),
            "language": language,
        }
    )
    return system, user


class Generator:
    def __init__(self, client: LLMClient, laws: dict[str, Law], max_article_chars: int | None = None):
        self.client = client
        self.laws = laws
        self.max_article_chars = max_article_chars

    def answer(
        self,
        question: str,
        lang: str,
        articles: Sequence[Article],
        citations: Sequence[Citation],
    ) -> Answer:
        model = f"{self.client.provider}:{self.client.model}"
        system, user = build_messages(question, lang, articles, self.laws, self.max_article_chars)
        try:
            raw = self.client.complete(system, user)
        except LLMError as exc:
            log.warning("falling back to extractive answer: %s", exc)
            return Answer(
                question, lang, "extractive", EXTRACTIVE_LEAD[lang], list(citations),
                notes=[LLM_FAILED[lang]],
            )

        if REFUSAL_TOKEN.search(raw):
            return Answer(question, lang, "generative", REFUSAL[lang], list(citations), refused=True, model=model)

        checked = check_answer(raw, articles)
        notes = []
        if checked.invalid:
            notes.append(REMOVED[lang].format(items=", ".join(checked.invalid)))
        if checked.dropped:
            notes.append(DROPPED[lang].format(count=checked.dropped))
        details = dict(
            removed_citations=checked.invalid,
            dropped_sentences=checked.dropped,
            uncited_sentences=checked.uncited,
        )
        if not checked.valid:
            notes.append(WITHHELD[lang])
            return Answer(
                question, lang, "generative", REFUSAL[lang], list(citations),
                refused=True, model=model, notes=notes, **details,
            )

        if detect_language(CITATION.sub("", checked.text)) != lang:
            log.warning("withheld an answer in the wrong language")
            return Answer(
                question, lang, "extractive", EXTRACTIVE_LEAD[lang], list(citations),
                notes=[WRONG_LANGUAGE[lang]], **details,
            )

        if checked.uncited:
            notes.append(UNCITED[lang])
        order = {article_id: i for i, article_id in enumerate(checked.valid)}
        ordered = sorted(
            (replace(c, cited=c.article_id in order) for c in citations),
            key=lambda c: (order.get(c.article_id, len(order)), c.rank),
        )
        return Answer(question, lang, "generative", checked.text, ordered, model=model, notes=notes, **details)
