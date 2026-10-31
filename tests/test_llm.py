import json
from datetime import datetime, timedelta, timezone

import httpx
import pytest

from qanun.answer.llm import (
    PRESETS,
    AnthropicClient,
    ConfigError,
    LLMError,
    OpenAIClient,
    OpenAICompatibleClient,
    RateLimited,
    client_from_env,
    fast_client_from_env,
    model_from_env,
    request_options,
    strip_reasoning,
)
from qanun.answer.ratelimit import (
    RateLimiter,
    parse_duration,
    parse_reset_time,
    parse_retry_after,
)

SECRET = "test-key-that-must-not-leak"
OK = {"choices": [{"message": {"content": "hello"}, "finish_reason": "stop"}]}


class FakeTime:
    def __init__(self):
        self.now = 1000.0
        self.sleeps = []

    def clock(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


def limiter(fake, **kw):
    return RateLimiter(sleep=fake.sleep, clock=fake.clock, **kw)


def client(handler, lim=None, provider="compatible", model="m"):
    return OpenAICompatibleClient(
        SECRET, model, "https://llm.test/v1", provider, limiter=lim,
        transport=httpx.MockTransport(handler),
    )


# ---------------------------------------------------------------- parsing


@pytest.mark.parametrize(
    "value, seconds",
    [("7.66s", 7.66), ("2m59.56s", 179.56), ("1h2m3s", 3723.0), ("120ms", 0.12), ("2", 2.0),
     ("", None), (None, None), ("soon", None), ("5x", None)],
)
def test_parse_duration(value, seconds):
    assert parse_duration(value) == (pytest.approx(seconds) if seconds is not None else None)


def test_parse_retry_after_handles_seconds_and_dates():
    assert parse_retry_after("2") == 2.0
    now = datetime(2026, 10, 31, 12, 0, tzinfo=timezone.utc)
    later = (now + timedelta(seconds=30)).strftime("%a, %d %b %Y %H:%M:%S GMT")
    assert parse_retry_after(later, now) == pytest.approx(30)
    assert parse_retry_after("not a date") is None


def test_parse_reset_time_handles_seconds_and_milliseconds():
    assert parse_reset_time("1800000060", now=1800000000) == pytest.approx(60)
    assert parse_reset_time("1800000060000", now=1800000000) == pytest.approx(60)
    assert parse_reset_time("nonsense") is None


# ---------------------------------------------------------------- limiter


def test_429_backs_off_for_retry_after_then_succeeds():
    fake = FakeTime()
    replies = [httpx.Response(429, headers={"retry-after": "3"}), httpx.Response(200, json=OK)]
    llm = client(lambda request: replies.pop(0), limiter(fake))
    assert llm.complete("s", "u") == "hello"
    assert fake.sleeps == [3.0]


def test_429_without_a_hint_backs_off_exponentially():
    fake = FakeTime()
    replies = [httpx.Response(429), httpx.Response(429), httpx.Response(200, json=OK)]
    llm = client(lambda request: replies.pop(0), limiter(fake, base_delay=1.0))
    assert llm.complete("s", "u") == "hello"
    assert 1.0 <= fake.sleeps[0] <= 1.25
    assert 2.0 <= fake.sleeps[1] <= 2.5


def test_a_long_wait_stops_instead_of_blocking():
    fake = FakeTime()
    llm = client(
        lambda request: httpx.Response(429, headers={"retry-after": "3600"}),
        limiter(fake, max_wait=30),
    )
    with pytest.raises(RateLimited) as caught:
        llm.complete("s", "u")
    assert caught.value.retry_after == 3600
    assert fake.sleeps == []


def test_retries_are_capped():
    fake = FakeTime()
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(429, headers={"retry-after": "1"})

    with pytest.raises(RateLimited):
        client(handler, limiter(fake, max_retries=2)).complete("s", "u")
    assert len(calls) == 3


def test_no_request_is_sent_when_the_daily_limit_is_known_to_be_spent():
    fake = FakeTime()
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(200, json=OK, headers={
            "x-ratelimit-remaining-requests": "0", "x-ratelimit-reset-requests": "2h0m0s",
        })

    llm = client(handler, limiter(fake, max_wait=30))
    llm.complete("s", "u")
    with pytest.raises(RateLimited, match="request limit"):
        llm.complete("s", "u")
    assert len(calls) == 1


def test_waits_for_the_token_window_before_sending():
    fake = FakeTime()
    headers = {"x-ratelimit-remaining-tokens": "10", "x-ratelimit-reset-tokens": "7.5s"}
    llm = client(lambda request: httpx.Response(200, json=OK, headers=headers), limiter(fake))
    llm.complete("s", "u")
    llm.complete("s", "u")
    assert fake.sleeps == [pytest.approx(7.5)]


def test_counters_clear_once_their_window_has_passed():
    fake = FakeTime()
    lim = limiter(fake)
    lim.observe({"x-ratelimit-remaining-tokens": "0", "x-ratelimit-reset-tokens": "5s"})
    assert lim.wait_needed(100)[0] == pytest.approx(5)
    fake.now += 6
    assert lim.wait_needed(100)[0] == 0


def test_limits_are_read_from_headers():
    lim = limiter(FakeTime())
    lim.observe({"x-ratelimit-limit-requests": "1000", "x-ratelimit-limit-tokens": "8000"})
    assert (lim.limit_requests, lim.limit_tokens) == (1000, 8000)


def test_a_rate_limit_never_switches_model_or_key():
    seen = []

    def handler(request):
        seen.append((json.loads(request.content)["model"], request.headers["authorization"]))
        return httpx.Response(429, headers={"retry-after": "9999"})

    llm = client_from_env(
        {"QANUN_LLM_API_KEY": SECRET, "QANUN_LLM_FAST_MODEL": "cheap"},
        transport=httpx.MockTransport(handler),
    )
    with pytest.raises(RateLimited):
        llm.complete("s", "u")
    assert seen == [("openai/gpt-oss-120b", f"Bearer {SECRET}")]


# ---------------------------------------------------------------- replies


def test_think_blocks_are_stripped_from_the_reply():
    text = "<think>maybe [labour-2004:999]\nNOT_COVERED</think>\nThe answer [labour-2004:54]."
    assert strip_reasoning(text) == "The answer [labour-2004:54]."
    reply = {"choices": [{"message": {"content": text}, "finish_reason": "stop"}]}
    assert client(lambda r: httpx.Response(200, json=reply)).complete("s", "u") == strip_reasoning(text)


def test_an_unclosed_think_block_is_an_error_not_an_answer():
    reply = {"choices": [{"message": {"content": "<think>never finished"}, "finish_reason": "stop"}]}
    with pytest.raises(LLMError, match="empty reply"):
        client(lambda r: httpx.Response(200, json=reply)).complete("s", "u")


@pytest.mark.parametrize("content", ["", None])
def test_an_empty_reply_is_an_error_not_a_refusal(content):
    reply = {"choices": [{"message": {"content": content, "reasoning": "long thoughts"}, "finish_reason": "stop"}]}
    with pytest.raises(LLMError, match="empty reply"):
        client(lambda r: httpx.Response(200, json=reply)).complete("s", "u")


def test_a_cut_off_reply_is_an_error():
    reply = {"choices": [{"message": {"content": "half an ans"}, "finish_reason": "length"}]}
    with pytest.raises(LLMError, match="cut off"):
        client(lambda r: httpx.Response(200, json=reply)).complete("s", "u")


def test_usage_is_recorded():
    reply = {**OK, "usage": {"total_tokens": 42}}
    llm = client(lambda r: httpx.Response(200, json=reply))
    llm.complete("s", "u")
    assert llm.last_usage["total_tokens"] == 42


def test_errors_never_contain_the_key_or_the_provider_message():
    body = {"error": {"code": "invalid_api_key", "message": f"bad key {SECRET}"}}
    llm = client(lambda request: httpx.Response(401, json=body))
    with pytest.raises(LLMError) as caught:
        llm.complete("s", "u")
    assert SECRET not in str(caught.value)
    assert "401" in str(caught.value) and "invalid_api_key" in str(caught.value)


# ---------------------------------------------------------------- configuration


@pytest.mark.parametrize("name", sorted(PRESETS))
def test_presets_build_a_client(name):
    llm = client_from_env({"QANUN_LLM_PROVIDER": name, "QANUN_LLM_API_KEY": SECRET})
    assert isinstance(llm, OpenAICompatibleClient)
    assert llm.provider == name
    assert llm.model == PRESETS[name].model
    assert str(llm._client.base_url).rstrip("/") == PRESETS[name].base_url


def test_groq_is_the_default_for_a_bare_key():
    llm = client_from_env({"QANUN_LLM_API_KEY": SECRET})
    assert llm.provider == "groq"
    assert llm.model == "openai/gpt-oss-120b"
    assert llm.max_tokens_field == "max_completion_tokens"


def test_fast_model_is_used_for_cheap_steps():
    assert fast_client_from_env({"QANUN_LLM_API_KEY": SECRET}).model == "openai/gpt-oss-20b"
    env = {"QANUN_LLM_API_KEY": SECRET, "QANUN_LLM_FAST_MODEL": "other"}
    assert fast_client_from_env(env).model == "other"
    # Without a preset or a fast model, the answer model is used.
    env = {"QANUN_LLM_API_KEY": SECRET, "QANUN_LLM_BASE_URL": "https://x.test/v1", "QANUN_LLM_MODEL": "m"}
    assert fast_client_from_env(env).model == "m"


def test_each_client_has_its_own_limiter():
    env = {"QANUN_LLM_API_KEY": SECRET}
    assert client_from_env(env).limiter is not fast_client_from_env(env).limiter


def test_generic_server_from_base_url():
    llm = client_from_env({
        "QANUN_LLM_API_KEY": SECRET,
        "QANUN_LLM_BASE_URL": "https://my.server/v1/",
        "QANUN_LLM_MODEL": "local-model",
    })
    assert llm.provider == "compatible"
    assert llm.model == "local-model"
    assert str(llm._client.base_url).rstrip("/") == "https://my.server/v1"


def test_a_preset_is_recognised_from_its_base_url():
    llm = client_from_env({
        "QANUN_LLM_API_KEY": SECRET, "QANUN_LLM_BASE_URL": "https://api.groq.com/openai/v1",
    })
    assert llm.provider == "groq"


def test_copied_example_with_empty_key_means_no_llm():
    assert client_from_env({"QANUN_LLM_PROVIDER": "groq", "QANUN_LLM_API_KEY": ""}) is None


def test_compatible_needs_a_url_and_a_model():
    with pytest.raises(ConfigError, match="BASE_URL"):
        client_from_env({"QANUN_LLM_PROVIDER": "compatible", "QANUN_LLM_API_KEY": SECRET})
    with pytest.raises(ConfigError, match="QANUN_LLM_MODEL"):
        client_from_env({"QANUN_LLM_API_KEY": SECRET, "QANUN_LLM_BASE_URL": "https://x.test/v1"})


def test_existing_providers_still_resolve_first_when_their_key_is_the_only_one():
    assert isinstance(client_from_env({"OPENAI_API_KEY": SECRET}), OpenAIClient)
    llm = client_from_env({"ANTHROPIC_API_KEY": SECRET, "QANUN_LLM_MODEL": "m"})
    assert isinstance(llm, AnthropicClient)


def test_model_can_be_overridden_for_a_comparison():
    assert client_from_env({"QANUN_LLM_API_KEY": SECRET}, model="allam-2-7b").model == "allam-2-7b"


def test_model_from_env_needs_no_key():
    assert model_from_env({"QANUN_LLM_PROVIDER": "groq"}) == "openai/gpt-oss-120b"
    assert model_from_env({"QANUN_LLM_PROVIDER": "groq"}, fast=True) == "openai/gpt-oss-20b"


# ---------------------------------------------------------------- request shape


def test_groq_request_keeps_reasoning_out_of_the_reply():
    seen = {}

    def handler(request):
        seen.update(json.loads(request.content))
        seen["url"] = str(request.url)
        return httpx.Response(200, json=OK)

    llm = client_from_env({"QANUN_LLM_API_KEY": SECRET}, transport=httpx.MockTransport(handler))
    llm.complete("s", "u")
    assert seen["url"] == "https://api.groq.com/openai/v1/chat/completions"
    assert seen["include_reasoning"] is False
    assert seen["max_completion_tokens"] == 1500
    assert "max_tokens" not in seen


def test_reasoning_options_by_model():
    assert request_options("groq", "qwen/qwen3.8-27b") == {"reasoning_format": "hidden"}
    assert request_options("groq", "allam-2-7b") == {}
    assert request_options("openrouter", "openai/gpt-oss-120b") == {}
