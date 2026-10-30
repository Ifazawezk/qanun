"""Records stored in the corpus files."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Amendment:
    """An instrument that changed an article, for example Decree-Law No. 18 of 2020."""

    instrument_ar: str
    instrument_en: str | None
    instrument_article: str | None
    effective_date: str | None
    change: str = "amended"  # "amended" or "added"


@dataclass
class PreviousVersion:
    text_ar: str
    effective_from: str | None
    effective_to: str | None


@dataclass
class Article:
    id: str  # "<law key>:<label>", for example "labour-2004:52-bis"
    law: str
    label: str  # "54", "52-bis", "115-bis-1", "issuance-2"
    number: int
    kind: str  # "article" or "issuance"
    title_en: str
    title_ar: str
    chapter_en: str | None
    chapter_ar: str | None
    section_en: str | None
    section_ar: str | None
    text_ar: str
    text_en: str | None
    text_en_source: str | None  # "almeezan", "almeezan-legacy" or None
    status: str  # "in_force" or "repealed"
    amended: bool
    added: bool
    effective_date: str | None
    order: int
    source_url: str
    almeezan_id: str
    amendments: list[Amendment] = field(default_factory=list)
    history: list[PreviousVersion] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Article":
        data = dict(data)
        data["amendments"] = [Amendment(**a) for a in data.get("amendments", [])]
        data["history"] = [PreviousVersion(**h) for h in data.get("history", [])]
        return cls(**data)

    @property
    def in_force(self) -> bool:
        return self.status == "in_force"

    def text(self, lang: str) -> str:
        """Text in the requested language, falling back to Arabic."""
        if lang == "en" and self.text_en:
            return self.text_en
        return self.text_ar


@dataclass
class Law:
    key: str
    name_en: str
    name_ar: str
    title_en: str | None
    title_ar: str
    number: int
    year: int
    type: str
    status: str
    publish_date: str | None
    effective_date: str | None
    gazette: str | None
    url: str
    almeezan_id: str
    article_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
