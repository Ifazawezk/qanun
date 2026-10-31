"""Builds the assistant from environment settings (shared by the CLI and the API)."""

from __future__ import annotations

import os

from qanun.answer.assistant import Assistant
from qanun.answer.generate import Generator
from qanun.answer.llm import client_from_env, fast_client_from_env
from qanun.answer.translate import LLMTranslator
from qanun.env import load_env_file
from qanun.retrieval.dense import DEFAULT_MODEL
from qanun.search import SearchEngine

OFF = ("", "none", "off", "false", "0")
DEFAULT_RERANKER = "mmarco"


def embedding_model(override: str | None = None) -> str | None:
    """QANUN_EMBEDDING_MODEL picks the model; "none" turns dense retrieval off."""
    value = override or os.environ.get("QANUN_EMBEDDING_MODEL", DEFAULT_MODEL)
    return None if value.lower() in OFF else value


def translation_enabled(settings=None) -> bool:
    settings = os.environ if settings is None else settings
    return settings.get("QANUN_TRANSLATE", "on").lower() not in OFF


def reranker_model(embedding: str | None, translating: bool, settings=None) -> str | None:
    """The reranker is on by default only alongside translation. That is where it
    measurably helps; without a translation its gain could be noise, so it would
    cost a model download and seconds per question for nothing shown.
    QANUN_RERANKER picks another model or turns it off."""
    settings = os.environ if settings is None else settings
    value = settings.get("QANUN_RERANKER")
    if value is None:
        return DEFAULT_RERANKER if embedding and translating else None
    return None if value.lower() in OFF else value


def build_assistant(model: str | None = None, env=None) -> Assistant:
    if env is None:
        load_env_file()
    client = client_from_env(env)
    translating = client is not None and translation_enabled(env)
    embedding = embedding_model(model)
    engine = SearchEngine.load(embedding, reranker_model(embedding, translating, env))
    generator = Generator(client, engine.laws) if client else None
    translator = LLMTranslator(fast_client_from_env(env)) if translating else None
    return Assistant(engine, generator, translator=translator)
