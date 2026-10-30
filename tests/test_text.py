import pytest

from qanun.text.normalize import detect_language, normalize, normalize_arabic, strip_diacritics
from qanun.text.tokenize import analyze, stem_arabic, tokens


def test_strip_diacritics_and_tatweel():
    assert strip_diacritics("يُعْمَلُ") == "يعمل"
    assert strip_diacritics("العـــامل") == "العامل"


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("أجر", "اجر"),
        ("إجازة", "اجازه"),
        ("آخر", "اخر"),
        ("ٱلعامل", "العامل"),
        ("مستشفى", "مستشفي"),
        ("مسؤول", "مسوول"),
        ("رئيس", "رييس"),
    ],
)
def test_normalize_arabic_letter_variants(raw, expected):
    assert normalize_arabic(raw) == expected


def test_normalize_arabic_options_can_be_turned_off():
    assert normalize_arabic("إجازة", teh_marbuta=False) == "اجازة"
    assert normalize_arabic("إجازة", alef=False) == "إجازه"


def test_normalize_mixed_text():
    assert normalize("Article ٥٤: مكافأة") == "article 54: مكافاه"
    assert normalize("كم ساعة؟") == "كم ساعه?"
    assert normalize("ﻻ") == "لا"  # presentation form ligature
    assert normalize("وافد‌للعمل") == "وافدللعمل"


def test_detect_language():
    assert detect_language("ما هي مدة الإجازة السنوية؟") == "ar"
    assert detect_language("How many days of annual leave?") == "en"
    assert detect_language("Article 54 مكافأة نهاية الخدمة") == "ar"
    assert detect_language("123") == "en"


@pytest.mark.parametrize(
    "word, stem",
    [
        ("العامل", "عامل"),
        ("للعامل", "عامل"),
        ("والعمال", "عمال"),
        ("العاملين", "عامل"),
        ("اجازات", "اجاز"),
        ("اجازه", "اجاز"),
        ("بالاجر", "اجر"),
    ],
)
def test_stem_arabic(word, stem):
    assert stem_arabic(word) == stem


def test_analyze_matches_spelling_variants():
    query = analyze("الأجازة السنوية")
    doc = analyze("إجازَةٌ سنويةٌ")
    assert query == doc


def test_analyze_drops_stopwords_and_stems_english():
    assert analyze("What is the notice period for workers?") == ["notic", "period", "worker"]
    assert analyze("ما هي مدة الإخطار") == ["مد", "اخطار"]


def test_tokens_keep_offsets_into_original_text():
    text = "يستحق العامل إجازةً سنوية"
    toks = tokens(text)
    assert [text[t.start : t.end] for t in toks] == ["يستحق", "العامل", "إجازةً", "سنوية"]
