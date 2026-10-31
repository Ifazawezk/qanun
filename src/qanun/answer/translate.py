"""Translates a question into the other language before searching.

17 Labour Law articles and every minimum wage text exist only in Arabic, so an
English question has no shared words with them. The translation is used only as
a search string: it is never shown as an answer or passed to the answer prompt.
A translator that fails or returns something unusable simply means the search
runs on the original question.
"""

from __future__ import annotations

import logging
import re
from typing import Protocol

from qanun.answer.generate import load_prompt
from qanun.answer.llm import LLMClient, LLMError
from qanun.text.normalize import detect_language

log = logging.getLogger("qanun.translate")

LANGUAGE_NAMES = {"en": "English", "ar": "Arabic"}
MAX_CHARS = 400


def other_language(lang: str) -> str:
    return "ar" if lang == "en" else "en"


class Translator(Protocol):
    def translate(self, question: str, lang: str) -> str | None: ...


def clean_translation(text: str, target: str) -> str | None:
    """Keep one short line in the target language, or nothing."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return None
    line = re.sub(r"\s+", " ", lines[0]).strip("\"'`«» ")
    if not line or len(line) > MAX_CHARS or detect_language(line) != target:
        return None
    return line


class LLMTranslator:
    def __init__(self, client: LLMClient):
        self.client = client

    def translate(self, question: str, lang: str) -> str | None:
        target = other_language(lang)
        system = load_prompt("translate").format_map({"language": LANGUAGE_NAMES[target]})
        try:
            reply = self.client.complete(system, f"<question>{question}</question>")
        except LLMError as exc:
            log.warning("searching without translation: %s", exc)
            return None
        return clean_translation(reply, target)
