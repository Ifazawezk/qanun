"""Shared test helpers.

FakeEncoder stands in for the sentence-transformers model so the tests run
offline and fast. It hashes normalized, stemmed terms into a bag-of-words
vector, plus a tiny hand-made bilingual lexicon so cross-lingual behaviour can
be tested without a real multilingual model.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from qanun.corpus.models import Amendment, Article, Law
from qanun.text.tokenize import analyze

LEXICON = {
    "اجاز": "leav",
    "سنوي": "annual",
    "مكافا": "gratuiti",
    "خدم": "servic",
    "رمضان": "ramadan",
    "ساع": "hour",
    "اخطار": "notic",
}


class FakeEncoder:
    name = "fake-encoder"
    max_chars = 400

    def __init__(self, dims: int = 256):
        self.dims = dims

    def _vector(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dims, dtype=np.float32)
        for term in analyze(text):
            term = LEXICON.get(term, term)
            slot = int(hashlib.md5(term.encode("utf-8")).hexdigest(), 16) % self.dims
            vec[slot] += 1.0
        norm = np.linalg.norm(vec)
        return vec / norm if norm else vec

    def encode_queries(self, texts):
        return np.stack([self._vector(t) for t in texts])

    def encode_passages(self, texts):
        return np.stack([self._vector(t) for t in texts])


def make_article(label: str, text_ar: str, text_en: str | None = None, law: str = "labour-2004", **kw) -> Article:
    number = int(label.split("-")[0]) if label[0].isdigit() else 1
    defaults = dict(
        id=f"{law}:{label}",
        law=law,
        label=label,
        number=number,
        kind="article",
        title_en=f"Article {label}",
        title_ar=f"المادة {label}",
        chapter_en="Chapter Seven",
        chapter_ar="الفصل السابع",
        section_en=None,
        section_ar=None,
        text_ar=text_ar,
        text_en=text_en,
        text_en_source="almeezan" if text_en else None,
        status="in_force",
        amended=False,
        added=False,
        effective_date="2005-01-02",
        order=number,
        source_url=f"https://almeezan.qa/legislations/x?almawadId={label}",
        almeezan_id=label,
    )
    defaults.update(kw)
    return Article(**defaults)


@pytest.fixture()
def fake_encoder():
    return FakeEncoder()


@pytest.fixture()
def sample_articles():
    return [
        make_article(
            "73",
            "يكون الحد الأقصى لساعات العمل العادية ثمان وأربعين ساعة في الأسبوع، عدا شهر رمضان فيكون ست وثلاثين ساعة.",
            "The maximum ordinary working hours shall be forty-eight hours per week, except in Ramadan.",
        ),
        make_article(
            "79",
            "يستحق العامل الذي أمضى سنة كاملة إجازة سنوية بأجر لا تقل عن ثلاثة أسابيع.",
            None,
        ),
        make_article(
            "54",
            "يجب على صاحب العمل أن يؤدي مكافأة نهاية خدمة للعامل الذي قضى في العمل مدة سنة كاملة فأكثر.",
            "The Employer shall pay the end of service gratuity to the Worker who spent one year or more.",
        ),
        make_article(
            "49",
            "يجوز لكل من طرفي عقد العمل إنهاء العقد بشرط الإخطار كتابة.",
            None,
            amended=True,
            amendments=[Amendment("مرسوم بقانون رقم (18) لسنة 2020", "Decree-Law No. 18 of 2020", "1", "2020-09-09")],
        ),
        make_article(
            "2",
            "يكون الحد الأدنى للأجر الذي يستحقه العامل ألف ريال شهريا.",
            None,
            law="min-wage-decision-2020",
        ),
        make_article("22", "نص ملغى", None, law="residency-2015", status="repealed"),
    ]


@pytest.fixture()
def sample_laws():
    def law(key, en, ar):
        return Law(
            key=key, name_en=en, name_ar=ar, title_en=en, title_ar=ar, number=1, year=2020,
            type="law", status="In Force", publish_date=None, effective_date=None, gazette=None,
            url=f"https://almeezan.qa/{key}", almeezan_id=key,
        )

    return {
        "labour-2004": law("labour-2004", "Labour Law", "قانون العمل"),
        "min-wage-decision-2020": law("min-wage-decision-2020", "Minimum wage decision", "قرار الحد الأدنى للأجور"),
        "residency-2015": law("residency-2015", "Residency Law", "قانون الإقامة"),
    }
