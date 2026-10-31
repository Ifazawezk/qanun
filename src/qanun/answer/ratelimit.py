"""Staying inside a provider's rate limits.

The limiter reads the limit headers on every response, waits before sending a
request it knows would be refused, and backs off on 429. When the wait would be
too long, such as a daily limit, it raises RateLimited and the caller stops.

It never works around a limit: there is one key and one model per client, and
nothing here switches key, account, model or provider.
"""

from __future__ import annotations

import logging
import random
import re
import threading
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Callable, Mapping

import httpx

log = logging.getLogger("qanun.ratelimit")

DURATION_PART = re.compile(r"(\d+(?:\.\d+)?)(ms|h|m|s)")
UNITS = {"h": 3600.0, "m": 60.0, "s": 1.0, "ms": 0.001}
RETRY_STATUSES = (429, 503)


class LLMError(RuntimeError):
    pass


class RateLimited(LLMError):
    """The provider's limit is reached and waiting it out here is not allowed."""

    def __init__(self, message: str, retry_after: float | None = None):
        super().__init__(message)
        self.retry_after = retry_after


def parse_duration(value: str | None) -> float | None:
    """Seconds from a reset value such as 7.66s, 2m59.56s, 1h2m3s or 120ms."""
    if not value:
        return None
    value = value.strip()
    try:
        return max(float(value), 0.0)
    except ValueError:
        pass
    parts = DURATION_PART.findall(value)
    if not parts or "".join(a + b for a, b in parts) != value:
        return None
    return sum(float(amount) * UNITS[unit] for amount, unit in parts)


def parse_retry_after(value: str | None, now: datetime | None = None) -> float | None:
    """Retry-After holds seconds or an HTTP date."""
    seconds = parse_duration(value)
    if seconds is not None or not value:
        return seconds
    try:
        moment = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    now = now or datetime.now(timezone.utc)
    return max((moment - now).total_seconds(), 0.0)


def parse_reset_time(value: str | None, now: float | None = None) -> float | None:
    """Seconds until X-RateLimit-Reset, sent as a Unix time in s or ms on a 429."""
    try:
        stamp = float(value) if value else None
    except ValueError:
        return None
    if stamp is None:
        return None
    if stamp > 1e11:
        stamp /= 1000.0
    now = time.time() if now is None else now
    return max(stamp - now, 0.0) if stamp > 1e9 else stamp


def _int(value: str | None) -> int | None:
    try:
        return int(float(value)) if value else None
    except ValueError:
        return None


class RateLimiter:
    def __init__(
        self,
        max_retries: int = 4,
        max_wait: float = 30.0,
        base_delay: float = 2.0,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.max_retries = max_retries
        self.max_wait = max_wait
        self.base_delay = base_delay
        self.sleep = sleep
        self.clock = clock
        self._lock = threading.Lock()
        self.limit_requests: int | None = None
        self.limit_tokens: int | None = None
        self.remaining_requests: int | None = None
        self.remaining_tokens: int | None = None
        self._requests_reset_at: float | None = None
        self._tokens_reset_at: float | None = None

    def observe(self, headers: Mapping[str, str]) -> None:
        """Remember the limits a response reports. Headers that are absent change nothing."""
        now = self.clock()
        limit_requests = _int(headers.get("x-ratelimit-limit-requests"))
        limit_tokens = _int(headers.get("x-ratelimit-limit-tokens"))
        remaining_requests = _int(headers.get("x-ratelimit-remaining-requests"))
        remaining_tokens = _int(headers.get("x-ratelimit-remaining-tokens"))
        reset_requests = parse_duration(headers.get("x-ratelimit-reset-requests"))
        reset_tokens = parse_duration(headers.get("x-ratelimit-reset-tokens"))
        with self._lock:
            if limit_requests is not None:
                self.limit_requests = limit_requests
            if limit_tokens is not None:
                self.limit_tokens = limit_tokens
            if remaining_requests is not None:
                self.remaining_requests = remaining_requests
                self._requests_reset_at = None if reset_requests is None else now + reset_requests
            if remaining_tokens is not None:
                self.remaining_tokens = remaining_tokens
                self._tokens_reset_at = None if reset_tokens is None else now + reset_tokens

    def wait_needed(self, estimated_tokens: int = 0) -> tuple[float, str]:
        """How long to hold a request back so that it will not be refused."""
        now = self.clock()
        wait, reason = 0.0, ""
        with self._lock:
            if self._requests_reset_at is not None and self._requests_reset_at <= now:
                self.remaining_requests, self._requests_reset_at = None, None
            if self._tokens_reset_at is not None and self._tokens_reset_at <= now:
                self.remaining_tokens, self._tokens_reset_at = None, None
            if self.remaining_requests == 0:
                wait, reason = (self._requests_reset_at or now) - now, "request limit"
            if self.remaining_tokens is not None and self.remaining_tokens < estimated_tokens:
                token_wait = (self._tokens_reset_at or now) - now
                if token_wait > wait:
                    wait, reason = token_wait, "token limit"
        return wait, reason

    def retry_delay(self, response: httpx.Response, attempt: int) -> float:
        headers = response.headers
        hinted = parse_retry_after(headers.get("retry-after"))
        if hinted is None:
            hinted = parse_reset_time(headers.get("x-ratelimit-reset"))
        if hinted is not None:
            return hinted
        # Jitter keeps parallel workers from retrying in step.
        return self.base_delay * (2**attempt) * (1 + random.random() / 4)

    def send(self, call: Callable[[], httpx.Response], estimated_tokens: int = 0) -> httpx.Response:
        """Run an HTTP call, waiting for the limits and retrying on 429 or 503."""
        attempt = 0
        while True:
            wait, reason = self.wait_needed(estimated_tokens)
            if wait > self.max_wait:
                raise RateLimited(f"{reason} reached; it resets in {wait:.0f}s", wait)
            if wait > 0:
                log.info("waiting %.1fs for the %s", wait, reason)
                self.sleep(wait)

            response = call()
            self.observe(response.headers)
            if response.status_code not in RETRY_STATUSES:
                return response

            delay = self.retry_delay(response, attempt)
            if attempt >= self.max_retries or delay > self.max_wait:
                raise RateLimited(
                    f"provider returned HTTP {response.status_code}; retry in {delay:.0f}s", delay
                )
            log.info("HTTP %d, backing off for %.1fs", response.status_code, delay)
            self.sleep(delay)
            attempt += 1
