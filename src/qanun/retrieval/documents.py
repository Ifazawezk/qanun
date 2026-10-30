"""How an article is turned into text for the indexes."""

from __future__ import annotations

from dataclasses import dataclass

from qanun.corpus.models import Article, Law


def headings(article: Article, lang: str) -> str:
    if lang == "ar":
        parts = [article.title_ar, article.chapter_ar, article.section_ar]
    else:
        parts = [article.title_en, article.chapter_en, article.section_en]
    return " - ".join(p for p in parts if p)


def article_document(article: Article, law: Law | None = None) -> str:
    """Everything BM25 should match on: both languages plus headings."""
    parts = [headings(article, "ar"), headings(article, "en")]
    if law is not None:
        parts += [law.name_ar, law.name_en]
    parts.append(article.text_ar)
    if article.text_en:
        parts.append(article.text_en)
    return "\n".join(p for p in parts if p)


@dataclass(frozen=True)
class Passage:
    article_id: str
    lang: str
    text: str


def split_paragraphs(text: str, max_chars: int) -> list[str]:
    """Pack paragraphs into chunks of at most max_chars (longer paragraphs are cut)."""
    chunks: list[str] = []
    current = ""
    for para in (p.strip() for p in text.split("\n")):
        if not para:
            continue
        while len(para) > max_chars:
            cut = para.rfind(" ", 0, max_chars)
            cut = cut if cut > max_chars // 2 else max_chars
            if current:
                chunks.append(current)
                current = ""
            chunks.append(para[:cut].strip())
            para = para[cut:].strip()
        if current and len(current) + 1 + len(para) > max_chars:
            chunks.append(current)
            current = para
        else:
            current = f"{current}\n{para}" if current else para
    if current:
        chunks.append(current)
    return chunks


def article_passages(
    article: Article, law: Law | None = None, max_chars: int = 900
) -> list[Passage]:
    """Passages for the embedding index, one set per available language.

    Each passage starts with the article title and headings so a chunk cut
    from the middle of a long article still says what it is about.
    """
    passages = []
    for lang, text in (("ar", article.text_ar), ("en", article.text_en)):
        if not text:
            continue
        prefix = headings(article, lang)
        if law is not None:
            prefix = f"{law.name_ar if lang == 'ar' else law.name_en} - {prefix}"
        for chunk in split_paragraphs(text, max_chars):
            passages.append(Passage(article.id, lang, f"{prefix}\n{chunk}"))
    return passages
