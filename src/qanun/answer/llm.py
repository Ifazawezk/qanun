"""Minimal clients for the OpenAI and Anthropic HTTP APIs.

Both are called over plain HTTP with httpx so the project does not depend on
either vendor SDK, and tests can swap in a mock transport.

Environment variables:
    QANUN_LLM_PROVIDER   openai | anthropic | none (default: picked from which key is set)
    QANUN_LLM_MODEL      model name; required for anthropic, defaults to gpt-4o-mini for openai
    OPENAI_API_KEY, OPENAI_BASE_URL        (the base URL also works with OpenAI-compatible servers)
    ANTHROPIC_API_KEY, ANTHROPIC_BASE_URL
    QANUN_LLM_TIMEOUT    seconds, default 60
    QANUN_LLM_MAX_TOKENS default 800
"""

from __future__ import annotations

import os
from typing import Mapping, Protocol

import httpx

DEFAULT_OPENAI_MODEL = "gpt-4o-mini"


class LLMError(RuntimeError):
    pass


class ConfigError(ValueError):
    pass


class LLMClient(Protocol):
    provider: str
    model: str

    def complete(self, system: str, user: str) -> str: ...


class OpenAIClient:
    provider = "openai"

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_OPENAI_MODEL,
        base_url: str = "https://api.openai.com/v1",
        timeout: float = 60.0,
        max_tokens: int = 800,
        transport: httpx.BaseTransport | None = None,
    ):
        self.model = model
        self.max_tokens = max_tokens
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            timeout=timeout,
            headers={"Authorization": f"Bearer {api_key}"},
            transport=transport,
        )

    def complete(self, system: str, user: str) -> str:
        payload = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": self.max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        try:
            resp = self._client.post("/chat/completions", json=payload)
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"] or ""
        except (httpx.HTTPError, KeyError, IndexError, ValueError) as exc:
            raise LLMError(f"OpenAI request failed: {exc}") from exc


class AnthropicClient:
    provider = "anthropic"
    api_version = "2023-06-01"

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str = "https://api.anthropic.com",
        timeout: float = 60.0,
        max_tokens: int = 800,
        transport: httpx.BaseTransport | None = None,
    ):
        self.model = model
        self.max_tokens = max_tokens
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            timeout=timeout,
            headers={"x-api-key": api_key, "anthropic-version": self.api_version},
            transport=transport,
        )

    def complete(self, system: str, user: str) -> str:
        payload = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": 0,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }
        try:
            resp = self._client.post("/v1/messages", json=payload)
            resp.raise_for_status()
            blocks = resp.json()["content"]
            return "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
        except (httpx.HTTPError, KeyError, ValueError) as exc:
            raise LLMError(f"Anthropic request failed: {exc}") from exc


def client_from_env(
    env: Mapping[str, str] | None = None, transport: httpx.BaseTransport | None = None
) -> LLMClient | None:
    """Build a client from environment variables, or None when no key is set."""
    env = os.environ if env is None else env
    provider = env.get("QANUN_LLM_PROVIDER", "").strip().lower()
    if provider == "none":
        return None
    if not provider:
        if env.get("OPENAI_API_KEY"):
            provider = "openai"
        elif env.get("ANTHROPIC_API_KEY"):
            provider = "anthropic"
        else:
            return None

    model = env.get("QANUN_LLM_MODEL", "").strip()
    timeout = float(env.get("QANUN_LLM_TIMEOUT", "60"))
    max_tokens = int(env.get("QANUN_LLM_MAX_TOKENS", "800"))

    if provider == "openai":
        key = env.get("OPENAI_API_KEY")
        if not key:
            raise ConfigError("QANUN_LLM_PROVIDER=openai but OPENAI_API_KEY is not set")
        return OpenAIClient(
            key,
            model or DEFAULT_OPENAI_MODEL,
            env.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            timeout,
            max_tokens,
            transport,
        )
    if provider == "anthropic":
        key = env.get("ANTHROPIC_API_KEY")
        if not key:
            raise ConfigError("QANUN_LLM_PROVIDER=anthropic but ANTHROPIC_API_KEY is not set")
        if not model:
            raise ConfigError("set QANUN_LLM_MODEL to the Anthropic model id you want to use")
        return AnthropicClient(
            key,
            model,
            env.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com"),
            timeout,
            max_tokens,
            transport,
        )
    raise ConfigError(f"unknown QANUN_LLM_PROVIDER: {provider}")
