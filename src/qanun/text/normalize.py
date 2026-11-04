"""Text normalization for Arabic and English.

Arabic legal text is written with optional diacritics and several spellings of
the same letter (أ, إ, آ and ا; ة and ه; ى and ي). Queries typed on a phone
rarely match the official spelling, so both sides go through the same
normalization before matching.
"""

from __future__ import annotations

import re
import unicodedata

# Harakat, Quranic marks, superscript alef.
DIACRITICS = re.compile(
    "[ؐ-ًؚ-ٰٟۖ-ۜ۟-۪ۨ-ۭ]"
)
TATWEEL = "ـ"
ALEF_FORMS = re.compile("[آأإٱٲٳ]")  # آ أ إ ٱ ٲ ٳ
TEH_MARBUTA = "ة"  # ة
HEH = "ه"  # ه
ALEF_MAQSURA = "ى"  # ى
YEH = "ي"  # ي
WAW_HAMZA = "ؤ"  # ؤ
YEH_HAMZA = "ئ"  # ئ
DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")
PUNCTUATION = str.maketrans({"،": ",", "؛": ";", "؟": "?", "٪": "%"})
# Persian letters that look like Arabic ones and come from typing on a Persian
# layout. One article of the minimum wage law has a Persian yeh.
PERSIAN = str.maketrans(
    {"\N{ARABIC LETTER FARSI YEH}": "\N{ARABIC LETTER YEH}", "\N{ARABIC LETTER KEHEH}": "\N{ARABIC LETTER KAF}"}
)
INVISIBLE = re.compile("[​-‏‪-‮⁦-⁩﻿]")

ARABIC_CHAR = re.compile("[؀-ۿ]")
LATIN_CHAR = re.compile("[A-Za-z]")


def canonical(text: str) -> str:
    """NFC with Persian look-alike letters replaced. Diacritics are kept, so this is
    for text a model reads, such as passages for the embedding index. A shadda typed
    before a kasra, as in Article 116 of the Labour Law, is put back in order."""
    return unicodedata.normalize("NFC", text).translate(PERSIAN)


def strip_diacritics(text: str) -> str:
    return DIACRITICS.sub("", text).replace(TATWEEL, "")


def normalize_arabic(
    text: str,
    *,
    alef: bool = True,
    teh_marbuta: bool = True,
    alef_maqsura: bool = True,
    hamza: bool = True,
) -> str:
    """Remove diacritics and tatweel, and fold letter variants to one form."""
    text = strip_diacritics(text).translate(PERSIAN)
    if alef:
        text = ALEF_FORMS.sub("ا", text)
    if teh_marbuta:
        text = text.replace(TEH_MARBUTA, HEH)
    if alef_maqsura:
        text = text.replace(ALEF_MAQSURA, YEH)
    if hamza:
        text = text.replace(WAW_HAMZA, "و").replace(YEH_HAMZA, YEH)
    return text


def normalize(text: str) -> str:
    """Normalize mixed Arabic and English text for matching."""
    text = unicodedata.normalize("NFKC", text)
    text = INVISIBLE.sub("", text)
    text = text.translate(DIGITS).translate(PUNCTUATION)
    text = normalize_arabic(text)
    return text.lower()


def detect_language(text: str) -> str:
    """Return "ar" when Arabic letters dominate, otherwise "en"."""
    arabic = len(ARABIC_CHAR.findall(text))
    latin = len(LATIN_CHAR.findall(text))
    return "ar" if arabic and arabic >= latin else "en"
