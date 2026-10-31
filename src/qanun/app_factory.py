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


def embedding_model(override: str | None = None) -> str | None:
    """QANUN_EMBEDDING_MODEL picks the model; "none" turns dense retrieval off."""
    value = override or os.environ.get("QANUN_EMBEDDING_MODEL", DEFAULT_MODEL)
    return None if value.lower() in OFF else value


def reranker_model() -> str | None:
    """QANUN_RERANKER picks a cross-encoder. It is off unless set."""
    value = os.environ.get("QANUN_RERANKER", "")
    return None if value.lower() in OFF else value


def build_assistant(model: str | None = None, env=None) -> Assistant:
    if env is None:
        load_env_file()
    engine = SearchEngine.load(embedding_model(model), reranker_model())
    client = client_from_env(env)
    generator = Generator(client, engine.laws) if client else None
    settings = os.environ if env is None else env
    translate = settings.get("QANUN_TRANSLATE", "on").lower() not in OFF
    translator = LLMTranslator(fast_client_from_env(env)) if client and translate else None
    return Assistant(engine, generator, translator=translator)
