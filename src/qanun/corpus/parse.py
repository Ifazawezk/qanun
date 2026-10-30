"""Turn the raw Al Meezan responses into structured article records.

Usage:
    python -m qanun.corpus.parse     # writes data/corpus/articles.jsonl and laws.json
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from bs4 import BeautifulSoup

from qanun.config import CORPUS_DIR, RAW_DIR
from qanun.corpus.legacy_en import load_legacy_english
from qanun.corpus.models import Amendment, Article, Law, PreviousVersion
from qanun.corpus.sources import LAWS, LEGACY_ENGLISH_CUTOFF, PORTAL_BASE, LawSource

log = logging.getLogger("qanun.parse")

ARABIC_LETTER = re.compile(r"[ء-ي]")
INVISIBLE = re.compile(r"[​-‏‪-‮⁦-⁩﻿]")
ARTICLE_REF_AR = re.compile(r"\((\d{1,3})\)")
NUMBER = re.compile(r"\d+")
LATIN_LETTER = re.compile(r"[A-Za-z]")
RANGE_SUFFIX = re.compile(r"\s*\(\s*\d+\s*(?:-\s*\d+\s*)?\)?\s*$")
ARTICLE_TITLE = re.compile(
    r"مادة\s+(?P<number>\d+)\s*(?:\(\s*(?P<issuance>إصدار)\s*\)|\(\s*(?P<bis>مكرر\S*)\s*(?P<bis_n>\d+)?\s*\))?"
)
INSTRUMENT = re.compile(
    r"^(?P<kind>مرسوم بقانون|قانون|قرار[^()]*?)\s*رقم\s*\(?\s*(?P<number>\d+)\s*\)?\s*لسنة\s*(?P<year>\d{4})"
)
INSTRUMENT_KINDS_EN = {"مرسوم بقانون": "Decree-Law", "قانون": "Law"}


# ---------------------------------------------------------------- text helpers


def html_to_text(html: str | None) -> str:
    """Convert an article's HTML body into plain text with one paragraph per line."""
    if not html:
        return ""
    soup = BeautifulSoup(html, "lxml")
    for br in soup.find_all("br"):
        br.replace_with("\n")
    for block in soup.find_all(["p", "div", "li", "tr"]):
        block.append("\n")
    text = INVISIBLE.sub("", soup.get_text())
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines()]
    text = "\n".join(line for line in lines if line)
    return strip_quotes(text)


def strip_quotes(text: str) -> str:
    """Amended articles are quoted from the amending law: '"...text...".'"""
    text = text.strip()
    if text.startswith(('"', "“")):
        end = re.search(r"[\"”]\s*\.?\s*$", text)
        if end:
            return text[1 : end.start()].strip()
    return text


def is_english(text: str) -> bool:
    latin = len(LATIN_LETTER.findall(text))
    arabic = len(ARABIC_LETTER.findall(text))
    return latin > 0 and latin >= 4 * arabic


def reference_mismatch(text_ar: str, text_en: str) -> list[str]:
    """Numbers cited in brackets in the Arabic text that the English text never mentions."""
    english_numbers = set(NUMBER.findall(text_en))
    return sorted(set(ARTICLE_REF_AR.findall(text_ar)) - english_numbers, key=int)


def clean_heading(title: str | None) -> str | None:
    if not title:
        return None
    return RANGE_SUFFIX.sub("", title).strip() or None


def date_only(value: str | None) -> str | None:
    return value[:10] if value else None


# ---------------------------------------------------------------- labels


@dataclass(frozen=True)
class Label:
    number: int
    kind: str  # "article" or "issuance"
    bis: int | None  # None, 0 for plain "bis", n for "bis n"

    @property
    def slug(self) -> str:
        if self.kind == "issuance":
            return f"issuance-{self.number}"
        if self.bis is None:
            return str(self.number)
        return f"{self.number}-bis" + (f"-{self.bis}" if self.bis else "")

    @property
    def title_en(self) -> str:
        if self.kind == "issuance":
            return f"Issuance Article {self.number}"
        suffix = "" if self.bis is None else " bis" + (f" {self.bis}" if self.bis else "")
        return f"Article {self.number}{suffix}"

    @property
    def title_ar(self) -> str:
        if self.kind == "issuance":
            return f"المادة {self.number} من مواد الإصدار"
        suffix = "" if self.bis is None else " مكررًا" + (f" {self.bis}" if self.bis else "")
        return f"المادة {self.number}{suffix}"


def parse_label(title_ar: str) -> Label:
    match = ARTICLE_TITLE.search(title_ar)
    if not match:
        raise ValueError(f"cannot read article title: {title_ar!r}")
    number = int(match.group("number"))
    if match.group("issuance"):
        return Label(number, "issuance", None)
    if match.group("bis"):
        return Label(number, "article", int(match.group("bis_n") or 0))
    return Label(number, "article", None)


def is_canceled_title(title: str) -> bool:
    return "ملغاة" in title or "Canceled" in title


def instrument_names(title_ar: str) -> tuple[str, str | None]:
    """Short Arabic citation and its English form, e.g. 'Decree-Law No. 18 of 2020'."""
    match = INSTRUMENT.search(title_ar.strip())
    if not match:
        return title_ar.strip(), None
    kind_ar = match.group("kind").strip()
    kind_en = INSTRUMENT_KINDS_EN.get(kind_ar, "Decision")
    short_ar = f"{kind_ar} رقم ({match.group('number')}) لسنة {match.group('year')}"
    return short_ar, f"{kind_en} No. {match.group('number')} of {match.group('year')}"


# ---------------------------------------------------------------- raw access


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))["data"]


def walk_sections(
    sections: list[dict[str, Any]], chapter: dict | None = None
) -> Iterator[tuple[dict | None, dict | None, dict[str, Any]]]:
    """Yield (chapter, section, article) for every article in the tree."""
    for node in sections:
        top = chapter or node
        sub = node if chapter else None
        for article in node.get("almawad") or []:
            yield top, sub, article
        yield from walk_sections(node.get("sections") or [], top)


def _history_versions(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return _load(path).get("relatedAlmawad") or []


def _amendments(versions: list[dict[str, Any]]) -> list[Amendment]:
    """List the instruments that added or amended an article, oldest first.

    In Al Meezan's version chain an "Amendment" entry is an old version and
    points at the instrument that replaced it; an "Addition" entry points at
    the instrument that inserted the article into the law.
    """
    found: dict[tuple[str, str | None], Amendment] = {}
    for version in sorted(versions, key=lambda v: v.get("chainPosition") or 0):
        code = version.get("relationCode")
        title = version.get("toLegislationTitle")
        if code not in ("Amendment", "Addition") or not title or "تعديل" not in title:
            continue
        if code == "Amendment":
            change, date = "amended", date_only(version.get("cancelationDate"))
        else:
            change, date = "added", date_only(version.get("effectiveDate"))
        short_ar, short_en = instrument_names(title)
        article_no = version.get("toAlmadaNumber")
        key = (short_ar, str(article_no) if article_no is not None else None)
        if key not in found:
            found[key] = Amendment(
                instrument_ar=short_ar,
                instrument_en=short_en,
                instrument_article=key[1],
                effective_date=date,
                change=change,
            )
    return sorted(found.values(), key=lambda a: a.effective_date or "9999")


def _previous_versions(
    versions: list[dict[str, Any]], superseded: list[dict[str, Any]], current_text: str
) -> list[PreviousVersion]:
    out: list[PreviousVersion] = []
    seen = {current_text}
    candidates = [
        v for v in versions if v.get("relationCode") == "Amendment" and not v.get("isLatest")
    ] + superseded
    candidates.sort(key=lambda v: v.get("effectiveDate") or v.get("cancelationDate") or "")
    for version in candidates:
        text = html_to_text(version.get("description"))
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(
            PreviousVersion(
                text_ar=text,
                effective_from=date_only(version.get("effectiveDate")),
                effective_to=date_only(version.get("cancelationDate")),
            )
        )
    return out


# ---------------------------------------------------------------- law parsing


def parse_law(
    source: LawSource,
    raw_dir: Path = RAW_DIR,
    legacy: dict[tuple[str, int], tuple[str, list[str]]] | None = None,
) -> tuple[Law, list[Article]]:
    law_dir = raw_dir / source.key
    ar = _load(law_dir / "legislation.ar.json")
    en = _load(law_dir / "legislation.en.json")
    legislation_id = ar["id"]

    en_articles = {a["id"]: a for _, _, a in walk_sections(en["sections"])}
    en_sections = {}
    for chapter, section, _ in walk_sections(en["sections"]):
        for node in (chapter, section):
            if node:
                en_sections[node["id"]] = node["title"]

    # Group every entry by its label; a label can have a current version and
    # one or more superseded versions that Al Meezan marks as canceled.
    groups: dict[Label, list[tuple[dict | None, dict | None, dict]]] = {}
    for chapter, section, article in walk_sections(ar["sections"]):
        groups.setdefault(parse_label(article["title"]), []).append((chapter, section, article))

    articles: list[Article] = []
    for order, (label, entries) in enumerate(groups.items()):
        current = [e for e in entries if not is_canceled_title(e[2]["title"])]
        repealed = not current
        chapter, section, item = (current or entries)[-1]
        superseded = [e[2] for e in entries if e[2] is not item]

        text_ar = html_to_text(item.get("description"))
        notes: list[str] = []

        en_item = en_articles.get(item["id"], {})
        en_text = html_to_text(en_item.get("description"))
        text_en, text_en_source = None, None
        effective = date_only(item.get("effectiveDate"))
        if en_text and is_english(en_text):
            text_en, text_en_source = en_text, "almeezan"
        elif (
            legacy
            and label.bis is None
            and (label.kind, label.number) in legacy
            and effective is not None
            and effective < LEGACY_ENGLISH_CUTOFF
        ):
            text_en, legacy_notes = legacy[(label.kind, label.number)]
            text_en_source = "almeezan-legacy"
            notes.extend(legacy_notes)

        if text_en:
            missing = reference_mismatch(text_ar, text_en)
            if missing:
                notes.append(
                    "The Arabic text cites "
                    + ", ".join(f"({n})" for n in missing)
                    + " but the English translation does not; the Arabic text is authoritative."
                )

        versions = _history_versions(law_dir / "history" / f"{item['id']}.ar.json")
        added = bool(item.get("isAddition"))

        def heading(node: dict | None, lang: str) -> str | None:
            if not node:
                return None
            if lang == "ar":
                return clean_heading(node["title"])
            title = en_sections.get(node["id"])
            return clean_heading(title) if title and is_english(title) else None

        articles.append(
            Article(
                id=f"{source.key}:{label.slug}",
                law=source.key,
                label=label.slug,
                number=label.number,
                kind=label.kind,
                title_en=label.title_en,
                title_ar=label.title_ar,
                chapter_en=heading(chapter, "en"),
                chapter_ar=heading(chapter, "ar"),
                section_en=heading(section, "en"),
                section_ar=heading(section, "ar"),
                text_ar=text_ar,
                text_en=text_en,
                text_en_source=text_en_source,
                status="repealed" if repealed else "in_force",
                amended=bool(item.get("isAmended")),
                added=added,
                effective_date=effective,
                order=order,
                source_url=f"{PORTAL_BASE}/legislations/{legislation_id}?almawadId={item['id']}",
                almeezan_id=item["id"],
                amendments=_amendments(versions),
                history=_previous_versions(versions, superseded, text_ar),
                notes=notes,
            )
        )

    gazette = ar.get("officialGazette") or {}
    law = Law(
        key=source.key,
        name_en=source.name_en,
        name_ar=source.name_ar,
        title_en=en["title"] if is_english(en["title"]) else None,
        title_ar=ar["title"],
        number=source.number,
        year=source.year,
        type=source.type_code,
        status=en.get("status") or ar.get("status") or "",
        publish_date=date_only(ar.get("publishDate")),
        effective_date=date_only(ar.get("effectiveDate")),
        gazette=gazette.get("title"),
        url=f"{PORTAL_BASE}{ar['legislationURL']}",
        almeezan_id=legislation_id,
        article_count=sum(1 for a in articles if a.in_force),
    )
    return law, articles


def build_corpus(raw_dir: Path = RAW_DIR) -> tuple[list[Law], list[Article]]:
    laws: list[Law] = []
    articles: list[Article] = []
    for source in LAWS:
        legacy = None
        legacy_pdf = raw_dir / source.key / "legacy-en.pdf"
        if legacy_pdf.exists():
            legacy = load_legacy_english(legacy_pdf)
        law, law_articles = parse_law(source, raw_dir, legacy)
        laws.append(law)
        articles.extend(law_articles)
        log.info(
            "%s: %d articles (%d in force, %d with English)",
            source.key,
            len(law_articles),
            law.article_count,
            sum(1 for a in law_articles if a.text_en),
        )
    return laws, articles


def main(argv: list[str] | None = None) -> None:
    from qanun.corpus.store import save_corpus

    parser = argparse.ArgumentParser(description="Parse raw Al Meezan data into articles")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--out-dir", type=Path, default=CORPUS_DIR)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    laws, articles = build_corpus(args.raw_dir)
    save_corpus(laws, articles, args.out_dir)


if __name__ == "__main__":
    main()
