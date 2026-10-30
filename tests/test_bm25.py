import numpy as np
import pytest

from qanun.retrieval.base import top_hits
from qanun.retrieval.bm25 import BM25Index
from qanun.retrieval.documents import split_paragraphs

DOCS = {
    "a": "يستحق العامل إجازة سنوية بأجر لا تقل عن ثلاثة أسابيع",
    "b": "The employer shall pay an end of service gratuity of three weeks per year",
    "c": "يكون الحد الأقصى لساعات العمل العادية ثمان وأربعين ساعة في الأسبوع وفي شهر رمضان ست وثلاثين ساعة",
    "d": "The worker may terminate the contract after giving written notice",
}


@pytest.fixture()
def index():
    return BM25Index().fit(list(DOCS), list(DOCS.values()), ["x", "x", "y", "y"])


def test_arabic_query_with_different_spelling_finds_article(index):
    hits = index.search("الاجازه السنويه", k=2)
    assert hits[0].article_id == "a"


def test_english_query_is_stemmed(index):
    hits = index.search("gratuities paid by employers")
    assert hits[0].article_id == "b"


def test_only_matching_documents_are_returned(index):
    hits = index.search("notice", k=10)
    assert [h.article_id for h in hits] == ["d"]
    assert hits[0].rank == 1


def test_unknown_terms_return_nothing(index):
    assert index.search("zzzz") == []


def test_law_filter(index):
    hits = index.search("ساعات العمل رمضان", laws=["x"])
    assert all(h.article_id in {"a", "b"} for h in hits)


def test_scores_follow_bm25_formula():
    index = BM25Index(k1=1.2, b=0.75).fit(["1", "2"], ["leave leave pay", "pay"])
    scores = index.scores("leave")
    n, df = 2, 1
    idf = np.log(1 + (n - df + 0.5) / (df + 0.5))
    tf, dl, avgdl = 2, 3, 2.0  # document lengths are 3 and 1 terms
    expected = idf * tf * 2.2 / (tf + 1.2 * (1 - 0.75 + 0.75 * dl / avgdl))
    assert scores[0] == pytest.approx(expected)
    assert scores[1] == 0


def test_top_hits_orders_and_limits():
    hits = top_hits(["a", "b", "c"], np.array([0.1, 0.9, 0.5]), k=2)
    assert [(h.article_id, h.rank) for h in hits] == [("b", 1), ("c", 2)]


def test_split_paragraphs_respects_limit():
    text = "first paragraph\nsecond paragraph\n" + "word " * 100
    chunks = split_paragraphs(text, max_chars=120)
    assert all(len(c) <= 120 for c in chunks)
    assert chunks[0] == "first paragraph\nsecond paragraph"
    assert "".join(chunks).replace("\n", "").replace(" ", "") == text.replace("\n", "").replace(" ", "")
