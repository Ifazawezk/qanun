from pathlib import Path

import pytest

from qanun.corpus.legacy_en import parse_articles
from qanun.corpus.parse import (
    html_to_text,
    instrument_names,
    is_english,
    parse_label,
    parse_law,
    reference_mismatch,
)
from qanun.corpus.sources import law_by_key

FIXTURES = Path(__file__).parent / "fixtures" / "raw"


def test_html_to_text_keeps_paragraphs_and_drops_tags():
    html = "<div>\r\n\t&quot;first line<BR>second <a href='/x'>12</a> line&quot;.</div><p></p>"
    assert html_to_text(html) == "first line\nsecond 12 line"


def test_html_to_text_drops_unmatched_closing_quote():
    assert html_to_text("<div>text of the article.&quot;.</div>") == "text of the article."
    assert html_to_text('a "quoted" word') == 'a "quoted" word'


def test_html_to_text_removes_zero_width_characters():
    assert html_to_text("للوافد‌ ‌للعمل") == "للوافد للعمل"


@pytest.mark.parametrize(
    "title, slug, title_en",
    [
        ("مادة 54 ", "54", "Article 54"),
        ("مادة 49  معدلة", "49", "Article 49"),
        ("مادة 52 (مكرر)", "52-bis", "Article 52 bis"),
        ("مادة 115 (مكررًا 1) معدلة", "115-bis-1", "Article 115 bis 1"),
        ("مادة 145 (مكرر 1)", "145-bis-1", "Article 145 bis 1"),
        ("مادة 2 (إصدار) معدلة", "issuance-2", "Issuance Article 2"),
    ],
)
def test_parse_label(title, slug, title_en):
    label = parse_label(title)
    assert label.slug == slug
    assert label.title_en == title_en


def test_parse_label_rejects_unknown_titles():
    with pytest.raises(ValueError):
        parse_label("Chapter One")


def test_instrument_names():
    ar, en = instrument_names(
        "مرسوم بقانون رقم (18) لسنة 2020 بتعديل بعض أحكام قانون العمل الصادر بالقانون رقم (14) لسنة 2004"
    )
    assert ar == "مرسوم بقانون رقم (18) لسنة 2020"
    assert en == "Decree-Law No. 18 of 2020"
    assert instrument_names("قانون رقم (13) لسنة 2017 بتعديل")[1] == "Law No. 13 of 2017"


def test_is_english():
    assert is_english("The Worker shall be entitled to annual leave (72).")
    assert not is_english("يستحق العامل إجازة سنوية")
    assert not is_english("")


def test_reference_mismatch_flags_numbers_missing_from_translation():
    assert reference_mismatch("أحكام المادة (74) من هذا القانون", "Article 75 of this Law") == ["74"]
    assert reference_mismatch("المادة (72)", "Article 72") == []


@pytest.fixture(scope="module")
def parsed():
    law, articles = parse_law(law_by_key("labour-2004"), FIXTURES)
    return law, {a.label: a for a in articles}


def test_parse_law_metadata(parsed):
    law, articles = parsed
    assert law.key == "labour-2004"
    assert law.title_en == "Law No. 14 of 2004 on the promulgation of Labour Law"
    assert law.url == "https://almeezan.qa/legislations/law/2004/14/general"
    assert law.article_count == sum(1 for a in articles.values() if a.in_force)


def test_parse_law_labels_and_chapters(parsed):
    _, articles = parsed
    assert set(articles) == {"issuance-1", "issuance-3", "49", "51", "52-bis", "54", "78"}
    art = articles["54"]
    assert art.id == "labour-2004:54"
    assert art.chapter_en == "Chapter Four"
    assert art.section_en == "Individual Work Relationship"
    assert art.section_ar == "علاقة العمل الفردية"
    assert "مكافأة نهاية خدمة" in art.text_ar
    assert art.source_url.startswith("https://almeezan.qa/legislations/")


def test_repealed_and_superseded_versions(parsed):
    _, articles = parsed
    assert articles["issuance-1"].status == "repealed"
    art51 = articles["51"]
    assert art51.status == "in_force"
    assert len(art51.history) == 1
    assert art51.history[0].effective_to == "2020-09-09"


def test_amendment_notes(parsed):
    _, articles = parsed
    (amendment,) = articles["49"].amendments
    assert amendment.instrument_en == "Decree-Law No. 18 of 2020"
    assert amendment.effective_date == "2020-09-09"
    assert amendment.change == "amended"
    assert articles["52-bis"].added


def test_english_text_falls_back_to_legacy_only_for_unchanged_articles():
    legacy = {
        ("article", 54): ("The Employer shall pay the end of service gratuity.", []),
        ("article", 49): ("Old notice period text.", []),
    }
    _, articles = parse_law(law_by_key("labour-2004"), FIXTURES, legacy)
    by_label = {a.label: a for a in articles}
    assert by_label["54"].text_en_source == "almeezan-legacy"
    # Article 49 was rewritten in 2020, so the 2009 translation must not be used.
    assert by_label["49"].text_en is None


LEGACY_SAMPLE = """Issuance Articles
Article 1 - Introduction

The provisions of the Labour Law attached herewith shall come into force.

Hamad bin Khalifa Al-Thani, Emir of the State of Qatar
Chapter One
Definitions and General Provisions
Article 4
Any conditions contrary to this Law shall be   and void unless they are more
advantageous to the Worker.

Article 5 (Amended By Law 6/2009)
(As amended pursuant to Article 1 of Law No. 6 of 2009)
The Worker shall be entitled to:
1. First item that wraps
onto a second line.
2. Second item.
Please do not consider the material presented above Official
"""


def test_legacy_english_parser():
    articles = parse_articles(LEGACY_SAMPLE)
    assert set(articles) == {("issuance", 1), ("article", 4), ("article", 5)}
    text4, notes4 = articles[("article", 4)]
    assert "[null] and void" in text4
    assert notes4
    text5, _ = articles[("article", 5)]
    assert text5.splitlines() == [
        "The Worker shall be entitled to:",
        "1. First item that wraps onto a second line.",
        "2. Second item.",
    ]
