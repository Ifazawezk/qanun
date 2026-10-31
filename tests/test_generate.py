import json

import httpx
import pytest

from qanun.answer.assistant import Assistant
from qanun.answer.generate import (
    REFUSAL,
    Generator,
    build_messages,
    check_citations,
    load_prompt,
    status_line,
)
from qanun.answer.llm import (
    AnthropicClient,
    ConfigError,
    LLMError,
    OpenAIClient,
    client_from_env,
)
from qanun.retrieval.dense import DenseIndex
from qanun.search import SearchEngine


class FakeLLM:
    provider = "fake"
    model = "test-model"

    def __init__(self, reply="", error=None):
        self.reply = reply
        self.error = error
        self.calls = []

    def complete(self, system, user):
        self.calls.append((system, user))
        if self.error:
            raise self.error
        return self.reply


# ---------------------------------------------------------------- configuration


def test_no_key_means_no_client():
    assert client_from_env({}) is None
    assert client_from_env({"OPENAI_API_KEY": "k", "QANUN_LLM_PROVIDER": "none"}) is None


def test_openai_is_picked_from_its_key_with_default_model():
    client = client_from_env({"OPENAI_API_KEY": "k"})
    assert isinstance(client, OpenAIClient)
    assert client.model == "gpt-4o-mini"
    client = client_from_env({"OPENAI_API_KEY": "k", "QANUN_LLM_MODEL": "my-model"})
    assert client.model == "my-model"


def test_anthropic_needs_a_model_name():
    with pytest.raises(ConfigError, match="QANUN_LLM_MODEL"):
        client_from_env({"ANTHROPIC_API_KEY": "k"})
    client = client_from_env({"ANTHROPIC_API_KEY": "k", "QANUN_LLM_MODEL": "some-model"})
    assert isinstance(client, AnthropicClient)


def test_bad_provider_settings():
    with pytest.raises(ConfigError):
        client_from_env({"QANUN_LLM_PROVIDER": "openai"})
    with pytest.raises(ConfigError):
        client_from_env({"QANUN_LLM_PROVIDER": "other", "OPENAI_API_KEY": "k"})


# ---------------------------------------------------------------- HTTP clients


def test_openai_request_and_response():
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["auth"] = request.headers["authorization"]
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"choices": [{"message": {"content": "answer [x:1]"}}]})

    client = OpenAIClient("secret", "m1", base_url="https://llm.test/v1", transport=httpx.MockTransport(handler))
    assert client.complete("sys", "usr") == "answer [x:1]"
    assert seen["url"] == "https://llm.test/v1/chat/completions"
    assert seen["auth"] == "Bearer secret"
    assert seen["body"]["model"] == "m1"
    assert seen["body"]["temperature"] == 0
    assert seen["body"]["messages"] == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "usr"},
    ]


def test_anthropic_request_and_response():
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["headers"] = request.headers
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"content": [{"type": "text", "text": "part one "}, {"type": "text", "text": "part two"}]})

    client = AnthropicClient("secret", "m2", base_url="https://llm.test", transport=httpx.MockTransport(handler))
    assert client.complete("sys", "usr") == "part one part two"
    assert seen["url"] == "https://llm.test/v1/messages"
    assert seen["headers"]["x-api-key"] == "secret"
    assert seen["headers"]["anthropic-version"] == "2023-06-01"
    assert seen["body"]["system"] == "sys"
    assert seen["body"]["messages"] == [{"role": "user", "content": "usr"}]


@pytest.mark.parametrize("cls", [OpenAIClient, AnthropicClient])
def test_http_errors_become_llm_errors(cls):
    transport = httpx.MockTransport(lambda request: httpx.Response(500, json={"error": "boom"}))
    client = cls("secret", "m", transport=transport)
    with pytest.raises(LLMError):
        client.complete("sys", "usr")


# ---------------------------------------------------------------- prompts


def test_prompts_are_read_from_files():
    assert "NOT_COVERED" in load_prompt("system")
    assert "{question}" in load_prompt("user")


def test_build_messages(sample_articles, sample_laws):
    articles = [sample_articles[0], sample_articles[3]]
    system, user = build_messages("ما مدة الإخطار؟", "ar", articles, sample_laws)
    assert "Write the answer in Arabic" in system
    assert '<article id="labour-2004:73"' in user
    assert "English (unofficial translation):" in user  # article 73 has English
    assert "amended by Decree-Law No. 18 of 2020" in user  # article 49
    assert "<question>ما مدة الإخطار؟</question>" in user


def test_status_line(sample_articles):
    assert status_line(sample_articles[0]) == "in force"
    assert status_line(sample_articles[3]) == "in force, amended by Decree-Law No. 18 of 2020"


def test_check_citations_drops_unknown_ids():
    text, valid, invalid = check_citations(
        "Notice is one month [labour-2004:49]. Pay continues [labour-2004:50] [labour-2004:999].",
        {"labour-2004:49", "labour-2004:50"},
    )
    assert valid == ["labour-2004:49", "labour-2004:50"]
    assert invalid == ["labour-2004:999"]
    assert text == "Notice is one month [labour-2004:49]. Pay continues [labour-2004:50]."


# ---------------------------------------------------------------- generator


@pytest.fixture()
def engine(sample_articles, sample_laws, fake_encoder):
    dense = DenseIndex.build(sample_articles, fake_encoder, sample_laws)
    return SearchEngine(sample_articles, sample_laws.values(), dense)


def test_generative_answer_marks_and_orders_cited_articles(engine):
    llm = FakeLLM("Working hours are 36 per week in Ramadan [labour-2004:73].")
    assistant = Assistant(engine, Generator(llm, engine.laws), top_k=3)
    answer = assistant.ask("What are the working hours in Ramadan?")
    assert answer.mode == "generative"
    assert answer.model == "fake:test-model"
    assert not answer.refused
    assert answer.citations[0].article_id == "labour-2004:73"
    assert answer.citations[0].cited
    assert not any(c.cited for c in answer.citations[1:])
    system, user = llm.calls[0]
    assert "Write the answer in English" in system
    assert "labour-2004:73" in user


def test_not_covered_reply_becomes_refusal(engine):
    assistant = Assistant(engine, Generator(FakeLLM("NOT_COVERED"), engine.laws))
    answer = assistant.ask("ما هي عقوبة التهرب الضريبي؟")
    assert answer.refused
    assert answer.text == REFUSAL["ar"]
    assert answer.citations  # the closest articles are still shown


def test_answer_without_valid_citation_is_withheld(engine):
    assistant = Assistant(engine, Generator(FakeLLM("Workers get 30 days [labour-2004:999]."), engine.laws))
    answer = assistant.ask("annual leave")
    assert answer.refused
    assert any("labour-2004:999" in note for note in answer.notes)


def test_llm_failure_falls_back_to_extractive(engine):
    assistant = Assistant(engine, Generator(FakeLLM(error=LLMError("timeout")), engine.laws))
    answer = assistant.ask("annual leave")
    assert answer.mode == "extractive"
    assert answer.notes


def test_extractive_mode_can_be_forced_with_llm_configured(engine):
    llm = FakeLLM("unused")
    assistant = Assistant(engine, Generator(llm, engine.laws))
    assert assistant.ask("annual leave", mode="extractive").mode == "extractive"
    assert llm.calls == []
