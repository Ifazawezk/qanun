"""A resumable local cache and request budget for evaluation runs.

Free tiers allow few tokens a day, so evaluation runs in small steps. Every reply
is appended to a local file as soon as it arrives, which means a run that stops
for any reason loses nothing, and the next run only pays for what is missing.

This is for evaluation only. The server never uses it, so questions from real
users are not written to disk.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from qanun.answer.llm import LLMClient


class BudgetExhausted(RuntimeError):
    """This run has spent the requests or tokens it was allowed."""


@dataclass
class Budget:
    """Limits for one run. Replies from the cache cost nothing."""

    requests: int | None = None
    tokens: int | None = None
    requests_used: int = 0
    tokens_used: int = 0
    cache_hits: int = 0

    def check(self) -> None:
        if self.requests is not None and self.requests_used >= self.requests:
            raise BudgetExhausted(f"request budget of {self.requests} is spent")
        if self.tokens is not None and self.tokens_used >= self.tokens:
            raise BudgetExhausted(f"token budget of {self.tokens} is spent")


class ResponseCache:
    """Append-only JSONL file. A line cut short by a crash is skipped on load."""

    def __init__(self, path: Path):
        self.path = path
        self.entries: dict[str, dict] = {}
        if path.exists():
            with path.open(encoding="utf-8") as fh:
                for line in fh:
                    try:
                        row = json.loads(line)
                        self.entries[row["key"]] = row
                    except (ValueError, KeyError, TypeError):
                        continue

    @staticmethod
    def key(provider: str, model: str, system: str, user: str) -> str:
        text = "\x00".join((provider, model, system, user))
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def get(self, key: str) -> dict | None:
        return self.entries.get(key)

    def put(self, key: str, model: str, text: str, tokens: int) -> None:
        row = {"key": key, "model": model, "text": text, "tokens": tokens}
        self.entries[key] = row
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()


class CachedClient:
    """An LLM client that answers from the cache and spends from a shared budget."""

    def __init__(self, client: LLMClient, cache: ResponseCache, budget: Budget):
        self.client = client
        self.cache = cache
        self.budget = budget
        self.provider = client.provider
        self.model = client.model

    def complete(self, system: str, user: str) -> str:
        key = ResponseCache.key(self.provider, self.model, system, user)
        hit = self.cache.get(key)
        if hit is not None:
            self.budget.cache_hits += 1
            return hit["text"]
        self.budget.check()
        self.budget.requests_used += 1
        text = self.client.complete(system, user)
        usage = getattr(self.client, "last_usage", None) or {}
        tokens = int(usage.get("total_tokens") or 0)
        self.budget.tokens_used += tokens
        self.cache.put(key, self.model, text, tokens)
        return text
