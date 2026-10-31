"""Clients for OpenAI-compatible servers and the Anthropic API.

All of them use plain HTTP with httpx, so the project depends on no vendor SDK
and tests can swap in a mock transport.

Environment variables:
    QANUN_LLM_PROVIDER     groq | openrouter | gemini | mistral | compatible | openai |
                           anthropic | none (default: picked from which key is set)
    QANUN_LLM_API_KEY      key for a preset or a compatible server
    QANUN_LLM_BASE_URL     any OpenAI-compatible server; overrides the preset's URL
    QANUN_LLM_MODEL        model for answers
    QANUN_LLM_FAST_MODEL   model for cheap steps such as query translation
    OPENAI_API_KEY, OPENAI_BASE_URL
    ANTHROPIC_API_KEY, ANTHROPIC_BASE_URL
    QANUN_LLM_TIMEOUT      seconds, default 60
    QANUN_LLM_MAX_TOKENS   default 800, or the preset's value
    QANUN_LLM_MAX_WAIT     longest wait for a rate limit, seconds, default 30

Errors never carry a key or the provider's error text, only the status and code.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, Mapping, Protocol
from urllib.parse import urlsplit

import httpx

from qanun.answer.ratelimit import LLMError, RateLimited, RateLimiter

__all__ = [
    "AnthropicClient", "ConfigError", "LLMClient", "LLMError", "OpenAIClient",
    "OpenAICompatibleClient", "PRESETS", "Preset", "RateLimited", "client_from_env",
    "fast_client_from_env", "model_from_env", "strip_reasoning",
]

DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
THINK_BLOCK = re.compile(r"<think>.*?(</think>|\Z)", re.DOTALL | re.IGNORECASE)
ERROR_CODE = re.compile(r"^[\w.-]{1,64}$")


class ConfigError(ValueError):
    pass


class LLMClient(Protocol):
    provider: str
    model: str

    def complete(self, system: str, user: str) -> str: ...


@dataclass(frozen=True)
class Preset:
    """Base URLs and model names were checked against each provider's docs in October 2026."""

    base_url: str
    model: str
    fast_model: str
    max_tokens: int = 800
    max_tokens_field: str = "max_tokens"


PRESETS = {
    # Reasoning models spend part of the output allowance on thinking, hence 1500.
    "groq": Preset(
        "https://api.groq.com/openai/v1", "openai/gpt-oss-120b", "openai/gpt-oss-20b",
        max_tokens=1500, max_tokens_field="max_completion_tokens",
    ),
    "openrouter": Preset(
        "https://openrouter.ai/api/v1", "google/gemma-4-31b-it:free", "google/gemma-4-26b-a4b-it:free",
    ),
    "gemini": Preset(
        "https://generativelanguage.googleapis.com/v1beta/openai", "gemini-3.8-flash", "gemini-3.8-flash",
    ),
    "mistral": Preset("https://api.mistral.ai/v1", "mistral-large-latest", "mistral-small-latest"),
}


def request_options(provider: str, model: str) -> dict[str, Any]:
    """Keep model reasoning out of the reply. On Groq, GPT OSS models return it in
    a separate field, and other models default to <think> tags inside the content."""
    if provider != "groq":
        return {}
    if model.startswith("openai/gpt-oss"):
        return {"include_reasoning": False, "reasoning_effort": "low"}
    if model.startswith("qwen/"):
        return {"reasoning_format": "hidden"}
    return {}


def strip_reasoning(text: str) -> str:
    """Remove <think> blocks, which may hold draft claims and citations. An
    unclosed block means the reply was cut off while thinking, so it all goes."""
    return THINK_BLOCK.sub("", text).strip()


def error_code(response: httpx.Response) -> str:
    """A short code from the error body, if there is one. The message is left out
    because providers can echo part of the key or the prompt in it."""
    try:
        error = response.json().get("error")
        code = (error.get("code") or error.get("type")) if isinstance(error, dict) else None
    except (ValueError, AttributeError):
        return ""
    return f" ({code})" if isinstance(code, str) and ERROR_CODE.match(code) else ""


class OpenAICompatibleClient:
    """Any server that speaks the OpenAI chat completions API."""

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str,
        provider: str = "compatible",
        timeout: float = 60.0,
        max_tokens: int = 800,
        max_tokens_field: str = "max_tokens",
        limiter: RateLimiter | None = None,
        transport: httpx.BaseTransport | None = None,
    ):
        self.provider = provider
        self.model = model
        self.max_tokens = max_tokens
        self.max_tokens_field = max_tokens_field
        self.limiter = limiter or RateLimiter()
        self.last_usage: dict[str, int] = {}
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
            self.max_tokens_field: self.max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            **request_options(self.provider, self.model),
        }
        # Arabic runs near 3 characters per token, and the output counts too.
        estimated = (len(system) + len(user)) // 3 + self.max_tokens
        try:
            resp = self.limiter.send(
                lambda: self._client.post("/chat/completions", json=payload), estimated
            )
            if resp.is_error:
                raise LLMError(f"{self.provider} returned HTTP {resp.status_code}{error_code(resp)}")
            body = resp.json()
            choice = body["choices"][0]
            content = strip_reasoning(choice["message"].get("content") or "")
        except LLMError:
            raise
        except (httpx.HTTPError, KeyError, IndexError, ValueError, TypeError, AttributeError) as exc:
            raise LLMError(f"{self.provider} request failed: {type(exc).__name__}") from None
        self.last_usage = body.get("usage") or {}
        if choice.get("finish_reason") == "length":
            raise LLMError(f"{self.provider} reply was cut off at {self.max_tokens} tokens")
        if not content:
            raise LLMError(f"{self.provider} returned an empty reply")
        return content


class OpenAIClient(OpenAICompatibleClient):
    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_OPENAI_MODEL,
        base_url: str = "https://api.openai.com/v1",
        timeout: float = 60.0,
        max_tokens: int = 800,
        transport: httpx.BaseTransport | None = None,
        limiter: RateLimiter | None = None,
    ):
        super().__init__(
            api_key, model, base_url, "openai", timeout, max_tokens,
            limiter=limiter, transport=transport,
        )


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
        limiter: RateLimiter | None = None,
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.limiter = limiter or RateLimiter()
        self.last_usage: dict[str, int] = {}
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
            resp = self.limiter.send(lambda: self._client.post("/v1/messages", json=payload))
            if resp.is_error:
                raise LLMError(f"anthropic returned HTTP {resp.status_code}{error_code(resp)}")
            body = resp.json()
            blocks = body["content"]
            text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
        except LLMError:
            raise
        except (httpx.HTTPError, KeyError, ValueError, TypeError, AttributeError) as exc:
            raise LLMError(f"anthropic request failed: {type(exc).__name__}") from None
        usage = body.get("usage") or {}
        self.last_usage = {
            "total_tokens": usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
        }
        return text


def _preset_for_url(base_url: str) -> str | None:
    host = urlsplit(base_url).hostname
    for name, preset in PRESETS.items():
        if host and urlsplit(preset.base_url).hostname == host:
            return name
    return None


def _provider(env: Mapping[str, str]) -> str | None:
    provider = env.get("QANUN_LLM_PROVIDER", "").strip().lower()
    if provider:
        return None if provider == "none" else provider
    if env.get("QANUN_LLM_API_KEY"):
        # Groq is the free default, so the key alone is enough to get going.
        base_url = env.get("QANUN_LLM_BASE_URL", "").strip()
        return (_preset_for_url(base_url) or "compatible") if base_url else "groq"
    if env.get("OPENAI_API_KEY"):
        return "openai"
    if env.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    return None


def model_from_env(env: Mapping[str, str] | None = None, fast: bool = False) -> str:
    """The model that would be used, for status output. Needs no key."""
    env = os.environ if env is None else env
    preset = PRESETS.get(_provider(env) or "")
    answers = env.get("QANUN_LLM_MODEL", "").strip()
    if fast:
        return env.get("QANUN_LLM_FAST_MODEL", "").strip() or (preset.fast_model if preset else answers)
    return answers or (preset.model if preset else "")


def client_from_env(
    env: Mapping[str, str] | None = None,
    transport: httpx.BaseTransport | None = None,
    fast: bool = False,
    model: str | None = None,
    max_wait: float | None = None,
    max_output: int | None = None,
) -> LLMClient | None:
    """Build a client from environment variables, or None when no key is set.

    fast picks the cheap model, and model overrides both. max_output limits this
    client's replies, for a model with a small context window. Each client has its
    own limiter because Groq sets limits per model.
    """
    env = os.environ if env is None else env
    provider = _provider(env)
    if provider is None:
        return None

    chosen = model or model_from_env(env, fast)
    timeout = float(env.get("QANUN_LLM_TIMEOUT", "60"))
    tokens = str(max_output or "") or env.get("QANUN_LLM_MAX_TOKENS", "").strip()
    max_tokens = int(tokens or 800)
    wait = max_wait if max_wait is not None else float(env.get("QANUN_LLM_MAX_WAIT", "30"))
    limiter = RateLimiter(max_wait=wait)

    if provider == "openai":
        key = env.get("OPENAI_API_KEY")
        if not key:
            raise ConfigError("QANUN_LLM_PROVIDER=openai but OPENAI_API_KEY is not set")
        return OpenAIClient(
            key,
            chosen or DEFAULT_OPENAI_MODEL,
            env.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            timeout,
            max_tokens,
            transport,
            limiter,
        )
    if provider == "anthropic":
        key = env.get("ANTHROPIC_API_KEY")
        if not key:
            raise ConfigError("QANUN_LLM_PROVIDER=anthropic but ANTHROPIC_API_KEY is not set")
        if not chosen:
            raise ConfigError("set QANUN_LLM_MODEL to the Anthropic model id you want to use")
        return AnthropicClient(
            key,
            chosen,
            env.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com"),
            timeout,
            max_tokens,
            transport,
            limiter,
        )

    preset = PRESETS.get(provider)
    if preset is None and provider != "compatible":
        raise ConfigError(f"unknown QANUN_LLM_PROVIDER: {provider}")
    key = env.get("QANUN_LLM_API_KEY", "").strip()
    if not key:
        # A copied .env.example with an empty key means extractive mode, not a crash.
        return None
    base_url = env.get("QANUN_LLM_BASE_URL", "").strip() or (preset.base_url if preset else "")
    if not base_url:
        raise ConfigError("QANUN_LLM_PROVIDER=compatible needs QANUN_LLM_BASE_URL")
    if not chosen:
        raise ConfigError("set QANUN_LLM_MODEL to the model you want to use")
    return OpenAICompatibleClient(
        key,
        chosen,
        base_url,
        provider,
        timeout,
        int(tokens or (preset.max_tokens if preset else 800)),
        preset.max_tokens_field if preset else "max_tokens",
        limiter,
        transport,
    )


def fast_client_from_env(
    env: Mapping[str, str] | None = None, transport: httpx.BaseTransport | None = None
) -> LLMClient | None:
    """The client for cheap steps. It only falls back to the answer model when no
    fast model exists, never because of a rate limit."""
    return client_from_env(env, transport, fast=True)
