"""Tokenization and light stemming for BM25 and passage highlighting."""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

import snowballstemmer

from qanun.text.normalize import ARABIC_CHAR, normalize

# A word is a run of letters or digits plus any Arabic diacritics or tatweel
# inside it, so "يُعمل" stays one word.
WORD = re.compile(r"[\wؐ-ًؚ-ٰٟـ]+")

# Stopwords are listed in normalized form (ة -> ه, ى -> ي, no hamza on alef).
ARABIC_STOPWORDS = frozenset(
    """
    في من الي علي عن ان او ما لا لم لن هذا هذه ذلك تلك التي الذي الذين اللذين
    هو هي هم كان كانت يكون تكون كل اي اذا ثم قد مع بين عند غير بعد قبل حتي
    و ف ب ل ك به بها له لها منه منها فيه فيها عليه عليها اليه اليها ذات
    ولا وان او اما الا لكن بل وهو وهي كما مما عما فان بان لان ايضا
    هل كم كيف متي اين ماذا لماذا
    """.split()
)
ENGLISH_STOPWORDS = frozenset(
    """
    a an the and or of to in on at by for with from as is are was were be been
    being it its this that these those there their they them he she his her
    him i me my we our you your which who whom whose what when where why how
    do does did doing have has had having not no nor so than too very can
    will would shall should may might must if then into onto about over under
    any all each other such only own same both more most some up out also
    """.split()
)

ARABIC_PREFIXES = ("وال", "بال", "كال", "فال", "لل", "ال")
ARABIC_SUFFIXES = ("ها", "ان", "ات", "ون", "ين", "يه", "ه", "ي")

_english = snowballstemmer.stemmer("english")


@lru_cache(maxsize=65536)
def stem_arabic(word: str) -> str:
    """Light stemming in the style of Larkey's light10, on normalized text.

    Strips the conjunction و, the definite article and its attached
    prepositions, then common plural, dual and pronoun suffixes. Roots are
    not extracted; that tends to hurt precision on legal text.
    """
    if len(word) >= 4 and word.startswith("و"):
        word = word[1:]
    for prefix in ARABIC_PREFIXES:
        if word.startswith(prefix) and len(word) - len(prefix) >= 2:
            word = word[len(prefix):]
            break
    for suffix in ARABIC_SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= 2:
            word = word[: -len(suffix)]
    return word


@lru_cache(maxsize=65536)
def stem_english(word: str) -> str:
    return _english.stemWord(word)


@dataclass(frozen=True)
class Token:
    term: str  # normalized and stemmed
    start: int  # offsets into the original text
    end: int


def analyze_word(word: str) -> str | None:
    """Normalize and stem one word; None if it is a stopword."""
    norm = normalize(word)
    if not norm:
        return None
    if ARABIC_CHAR.search(norm):
        if norm in ARABIC_STOPWORDS:
            return None
        return stem_arabic(norm)
    if norm in ENGLISH_STOPWORDS:
        return None
    if norm.isdigit():
        return norm.lstrip("0") or "0"
    return stem_english(norm)


def tokens(text: str) -> list[Token]:
    out = []
    for match in WORD.finditer(text):
        term = analyze_word(match.group())
        if term:
            out.append(Token(term, match.start(), match.end()))
    return out


def analyze(text: str) -> list[str]:
    """The terms BM25 indexes for a text."""
    return [t.term for t in tokens(text)]
