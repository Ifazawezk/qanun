"""Checks on the processed corpus that ships with the repository."""

from collections import Counter

import pytest

from qanun.config import CORPUS_DIR
from qanun.corpus.store import load_articles, load_laws

pytestmark = pytest.mark.skipif(
    not (CORPUS_DIR / "articles.jsonl").exists(), reason="corpus not built"
)


@pytest.fixture(scope="module")
def articles():
    return load_articles()


def test_ids_are_unique(articles):
    counts = Counter(a.id for a in articles)
    assert [i for i, n in counts.items() if n > 1] == []


def test_labour_law_has_every_article(articles):
    numbers = {
        a.number
        for a in articles
        if a.law == "labour-2004" and a.kind == "article" and a.in_force
    }
    assert numbers == set(range(1, 149))


def test_every_article_has_arabic_text_and_source(articles):
    for article in articles:
        assert article.text_ar.strip(), article.id
        assert article.source_url.startswith("https://almeezan.qa/"), article.id


def test_english_text_is_marked_with_its_source(articles):
    for article in articles:
        assert (article.text_en is None) == (article.text_en_source is None), article.id


def test_laws_file_matches_articles(articles):
    laws = {law.key: law for law in load_laws()}
    assert set(laws) == {a.law for a in articles}
    for key, law in laws.items():
        in_force = sum(1 for a in articles if a.law == key and a.in_force)
        assert law.article_count == in_force


def test_odd_unicode_in_the_source_is_kept_but_normalised_for_search(articles):
    """Article 116 types a shadda before a kasra (not NFC), and Article 1 of the
    minimum wage law has a Persian yeh in مستوى. Both stay as published."""
    import unicodedata

    from qanun.retrieval.documents import article_passages
    from qanun.text.tokenize import analyze

    by_id = {a.id: a for a in articles}
    reordered, persian = by_id["labour-2004:116"], by_id["min-wage-2020:1"]
    assert not unicodedata.is_normalized("NFC", reordered.text_ar)
    assert "\N{ARABIC LETTER FARSI YEH}" in persian.text_ar

    for article in (reordered, persian):
        clean = unicodedata.normalize("NFC", article.text_ar).replace("\N{ARABIC LETTER FARSI YEH}", "\N{ARABIC LETTER YEH}")
        assert analyze(article.text_ar) == analyze(clean), article.id
        for passage in article_passages(article):
            assert unicodedata.is_normalized("NFC", passage.text), article.id
            assert "\N{ARABIC LETTER FARSI YEH}" not in passage.text, article.id
    assert set(analyze("مستوى")) <= set(analyze(persian.text_ar))
