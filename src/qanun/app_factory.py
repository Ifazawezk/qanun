"""Builds the assistant from environment settings (shared by the CLI and the API)."""

from __future__ import annotations

import os

from qanun.answer.assistant import Assistant
from qanun.retrieval.dense import DEFAULT_MODEL
from qanun.search import SearchEngine


def embedding_model(override: str | None = None) -> str | None:
    """QANUN_EMBEDDING_MODEL picks the model; "none" turns dense retrieval off."""
    value = override or os.environ.get("QANUN_EMBEDDING_MODEL", DEFAULT_MODEL)
    return None if value.lower() in ("", "none", "off") else value


def build_assistant(model: str | None = None) -> Assistant:
    engine = SearchEngine.load(embedding_model(model))
    return Assistant(engine)
