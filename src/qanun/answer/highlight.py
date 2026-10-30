"""Pick the most relevant sentences of an article and mark the matching words."""

from __future__ import annotations

import re
from typing import Callable, Sequence

from qanun.answer.models import Passage
from qanun.text.tokenize import analyze, tokens

# Sentence boundaries: line breaks, or ". ؛ ;" followed by a space. A dot right
# after a digit ("1. If the worker...") is a list number, not a sentence end.
SENTENCE_BREAK = re.compile(r"\n+|(?<=[^\d\s][.؛;])\s+")


def split_sentences(text: str, min_chars: int = 12) -> list[tuple[int, int]]:
    """Character spans of the sentences in text. Fragments shorter than
    min_chars (like a bare list number) are joined to the sentence after them."""
    spans: list[tuple[int, int]] = []
    start = 0
    for match in SENTENCE_BREAK.finditer(text):
        if match.start() > start:
            spans.append((start, match.start()))
        start = match.end()
    if start < len(text):
        spans.append((start, len(text)))

    merged: list[tuple[int, int]] = []
    pending: int | None = None
    for a, b in spans:
        if pending is not None:
            a, pending = pending, None
        if b - a < min_chars:
            pending = a
            continue
        merged.append((a, b))
    if pending is not None:
        if merged:
            merged[-1] = (merged[-1][0], len(text))
        else:
            merged.append((pending, len(text)))
    return merged


def term_highlights(text: str, query_terms: set[str]) -> list[tuple[int, int]]:
    return [(t.start, t.end) for t in tokens(text) if t.term in query_terms]


def best_passage(
    question: str,
    text: str,
    lang: str,
    similarity: Callable[[Sequence[str]], Sequence[float]] | None = None,
    max_sentences: int = 2,
) -> Passage:
    """Choose up to max_sentences adjacent sentences that best match the question.

    Lexical overlap with the question's stemmed terms drives the choice. When
    a similarity function is given (the embedding model), it is added in, which
    matters when the question and the article are in different languages and
    no words overlap.
    """
    spans = split_sentences(text)
    if not spans:
        return Passage(text=text, lang=lang)
    query_terms = set(analyze(question))
    sentences = [text[a:b] for a, b in spans]

    overlap = []
    for sentence in sentences:
        terms = set(analyze(sentence))
        overlap.append(len(terms & query_terms) / (len(query_terms) or 1))
    semantic = list(similarity(sentences)) if similarity else [0.0] * len(sentences)
    scores = [o + s for o, s in zip(overlap, semantic)]

    best = max(range(len(spans)), key=lambda i: scores[i])
    end = best + 1
    # Add following sentences while they are relevant too, or while the passage
    # is only a lead-in such as "in the following cases:".
    while end < len(spans) and end - best < max_sentences:
        lead_in = sentences[end - 1].rstrip().endswith(":")
        if overlap[end] > 0 or lead_in:
            end += 1
        else:
            break
    start_char, end_char = spans[best][0], spans[end - 1][1]
    passage = text[start_char:end_char].strip()
    return Passage(text=passage, lang=lang, highlights=term_highlights(passage, query_terms))


def render_marked(passage: Passage, open_mark: str = "**", close_mark: str = "**") -> str:
    """The passage with highlighted ranges wrapped in markers (for the CLI)."""
    out, last = [], 0
    for start, end in passage.highlights:
        out.append(passage.text[last:start])
        out.append(f"{open_mark}{passage.text[start:end]}{close_mark}")
        last = end
    out.append(passage.text[last:])
    return "".join(out)
