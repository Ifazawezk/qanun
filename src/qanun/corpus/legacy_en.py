"""Parse the older Al Meezan English export of the Labour Law (a PDF).

The export lists the four issuance articles first ("Article 2 - Introduction")
and then the 146 articles of the law itself, each starting on its own
"Article N" line, sometimes followed by "(Amended By ...)" tags.
"""

from __future__ import annotations

import re
from pathlib import Path

from pypdf import PdfReader

HEADING = re.compile(
    r"^Article\s+(\d+)(\s+-\s+Introduction)?((?:\s*\(Amended By [^)]*\))*)\s*$"
)
CHAPTER = re.compile(r"^(Chapter\s+\w+|Issuance Articles)\s*$")
LIST_ITEM = re.compile(r"^(\d+[\.\-)]|\d+\s+[A-Z]|[a-z][\.)]|-|•)\s*")
PAGE_NOISE = (
    "Please do not consider the material presented above Official",
    "Al Meezan - Qatary Legal Portal",
)
AMENDED_PURSUANT = re.compile(r"^\(As amended pursuant to .*\)$")
# The export lost the word "null" in the phrase "null and void". We put it
# back in brackets so the restored word is visible as an editorial fix.
NULL_GAP = re.compile(r"\b(be|deemed|considered)\s{2,}and void")
LEADING_GAP = re.compile(r"^\s{2,}and void", re.M)


def extract_text(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def _unwrap(lines: list[str]) -> str:
    """Join the PDF's hard-wrapped lines back into paragraphs."""
    paragraphs: list[str] = []
    current: list[str] = []
    for raw in lines:
        line = raw.strip()
        if not line:
            if current:
                paragraphs.append(" ".join(current))
                current = []
            continue
        if current and (LIST_ITEM.match(line) or current[-1].endswith(":")):
            paragraphs.append(" ".join(current))
            current = []
        current.append(line)
    if current:
        paragraphs.append(" ".join(current))
    return "\n".join(re.sub(r"[ \t]+", " ", p) for p in paragraphs)


def parse_articles(text: str) -> dict[tuple[str, int], tuple[str, list[str]]]:
    """Return {(kind, number): (text, notes)} where kind is "issuance" or "article"."""
    articles: dict[tuple[str, int], tuple[str, list[str]]] = {}
    key: tuple[str, int] | None = None
    body: list[str] = []

    def flush() -> None:
        if key is None:
            return
        notes: list[str] = []
        raw = "\n".join(body)
        fixed = NULL_GAP.sub(r"\1 [null] and void", raw)
        fixed = LEADING_GAP.sub("[null] and void", fixed)
        if fixed != raw:
            notes.append(
                "The legacy English export dropped the word 'null' in 'null and void'; "
                "it is restored here in brackets."
            )
        content = _unwrap(fixed.split("\n"))
        if content:
            articles[key] = (content, notes)

    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if stripped in PAGE_NOISE or AMENDED_PURSUANT.match(stripped):
            continue
        heading = HEADING.match(stripped)
        if heading:
            flush()
            kind = "issuance" if heading.group(2) else "article"
            key = (kind, int(heading.group(1)))
            body = []
            continue
        if CHAPTER.match(stripped):
            flush()
            key, body = None, []
            continue
        if stripped.startswith(("Hamad bin Khalifa Al-Thani", "Signed at Emiri Diwan", "Corresponding to")):
            flush()
            key, body = None, []
            continue
        if key is not None:
            body.append(line)
    flush()
    # Chapter subtitles ("Occupational Training") sit between the chapter line
    # and the first article, so they never end up inside an article body.
    return articles


def load_legacy_english(pdf_path: Path) -> dict[tuple[str, int], tuple[str, list[str]]]:
    return parse_articles(extract_text(pdf_path))
