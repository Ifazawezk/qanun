"""The legislation that makes up the corpus and where each piece comes from."""

from __future__ import annotations

from dataclasses import dataclass

# Al Meezan (almeezan.qa) is the legal portal of Qatar's Ministry of Justice.
# Its web app reads legislation from this public JSON API.
API_BASE = "https://api.moj.gov.qa/api/v1"
PORTAL_BASE = "https://almeezan.qa"


@dataclass(frozen=True)
class LawSource:
    key: str
    number: int
    year: int
    type_code: str
    name_en: str
    name_ar: str
    source_code: str = "general"

    @property
    def lookup_params(self) -> dict[str, str]:
        return {
            "number": str(self.number),
            "year": str(self.year),
            "typeCode": self.type_code,
            "sourceCode": self.source_code,
        }


LAWS: tuple[LawSource, ...] = (
    LawSource(
        key="labour-2004",
        number=14,
        year=2004,
        type_code="law",
        name_en="Labour Law (Law No. 14 of 2004)",
        name_ar="قانون العمل (القانون رقم 14 لسنة 2004)",
    ),
    LawSource(
        key="min-wage-2020",
        number=17,
        year=2020,
        type_code="law",
        name_en="Minimum Wage Law (Law No. 17 of 2020)",
        name_ar="قانون الحد الأدنى للأجور (القانون رقم 17 لسنة 2020)",
    ),
    LawSource(
        key="min-wage-decision-2020",
        number=25,
        year=2020,
        type_code="ministerial-decision",
        name_en="Ministerial Decision No. 25 of 2020 on the minimum wage",
        name_ar="القرار الوزاري رقم 25 لسنة 2020 بتحديد الحد الأدنى للأجور",
    ),
    LawSource(
        key="residency-2015",
        number=21,
        year=2015,
        type_code="law",
        name_en="Expatriates Entry, Exit and Residency Law (Law No. 21 of 2015)",
        name_ar="قانون تنظيم دخول وخروج الوافدين وإقامتهم (القانون رقم 21 لسنة 2015)",
    ),
)

DEFAULT_LAW = "labour-2004"

# The current Al Meezan site has English text for only part of the Labour Law.
# An older Al Meezan English export of the full law (amendments up to 2009) is
# mirrored by the Center for Migration and Refugee Studies at AUC. We use it
# only for articles whose text has not changed since that export.
LEGACY_ENGLISH = {
    "labour-2004": "https://www4.aucegypt.edu/CMRS/Files/Law_14_2004_3961%20(1).pdf",
}
LEGACY_ENGLISH_CUTOFF = "2010-01-01"


def law_by_key(key: str) -> LawSource:
    for law in LAWS:
        if law.key == key:
            return law
    raise KeyError(f"unknown law key: {key}")
