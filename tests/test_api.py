import pytest
from fastapi.testclient import TestClient

from qanun.answer.assistant import Assistant
from qanun.answer.generate import Generator
from qanun.api.app import create_app
from qanun.retrieval.dense import DenseIndex
from qanun.search import SearchEngine


class FakeLLM:
    provider = "fake"
    model = "test-model"

    def __init__(self, reply):
        self.reply = reply

    def complete(self, system, user):
        return self.reply


@pytest.fixture()
def engine(sample_articles, sample_laws, fake_encoder):
    dense = DenseIndex.build(sample_articles, fake_encoder, sample_laws)
    return SearchEngine(sample_articles, sample_laws.values(), dense)


@pytest.fixture()
def client(engine, tmp_path):
    app = create_app(Assistant(engine, top_k=3), static_dir=tmp_path)
    with TestClient(app) as c:
        yield c


@pytest.fixture()
def llm_client(engine, tmp_path):
    generator = Generator(FakeLLM("Ordinary hours are 48 a week, 36 in Ramadan [labour-2004:73]."), engine.laws)
    app = create_app(Assistant(engine, generator, top_k=3), static_dir=tmp_path)
    with TestClient(app) as c:
        yield c


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["articles"] == 5  # the repealed sample article is not counted
    assert body["search_methods"] == ["bm25", "dense", "hybrid", "cross"]
    assert body["default_method"] == "cross"
    assert body["generation"] is False


def test_laws(client):
    keys = {law["key"] for law in client.get("/laws").json()}
    assert "labour-2004" in keys


def test_ask_extractive(client):
    resp = client.post("/ask", json={"question": "What are the working hours in Ramadan?"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["mode"] == "extractive"
    assert body["language"] == "en"
    top = body["citations"][0]
    assert top["article_id"] == "labour-2004:73"
    assert top["passage"]["highlights"]
    assert top["url"].startswith("https://almeezan.qa/")


def test_ask_arabic(client):
    body = client.post("/ask", json={"question": "كم مدة الإجازة السنوية؟", "k": 2}).json()
    assert body["language"] == "ar"
    assert len(body["citations"]) == 2
    assert body["citations"][0]["title"] == "المادة 79"


def test_ask_generative_with_mocked_llm(llm_client):
    body = llm_client.post("/ask", json={"question": "What are the working hours in Ramadan?"}).json()
    assert body["mode"] == "generative"
    assert "[labour-2004:73]" in body["text"]
    assert body["citations"][0]["cited"] is True
    assert body["model"] == "fake:test-model"
    assert llm_client.get("/health").json()["llm"] == "fake:test-model"


def test_ask_validation(client):
    assert client.post("/ask", json={"question": ""}).status_code == 422
    assert client.post("/ask", json={"question": "x" * 1001}).status_code == 422
    assert client.post("/ask", json={"question": "annual leave", "k": 0}).status_code == 422
    assert client.post("/ask", json={"question": "annual leave", "mode": "generative"}).status_code == 400
    assert client.post("/ask", json={"question": "annual leave", "laws": ["nope"]}).status_code == 400


def test_search_methods(client):
    for method in ("bm25", "dense", "hybrid"):
        body = client.get("/search", params={"q": "gratuity", "method": method, "k": 3}).json()
        assert body["method"] == method
        assert body["hits"][0]["article_id"] == "labour-2004:54"
    assert client.get("/search", params={"q": "gratuity", "method": "magic"}).status_code == 400


def test_search_law_filter(client):
    body = client.get("/search", params={"q": "الأجر", "law": "min-wage-decision-2020"}).json()
    assert {h["law"] for h in body["hits"]} == {"min-wage-decision-2020"}


def test_get_article(client):
    body = client.get("/articles/49").json()
    assert body["id"] == "labour-2004:49"
    assert body["law_name_en"] == "Labour Law"
    assert body["amendments"][0]["instrument_en"] == "Decree-Law No. 18 of 2020"
    assert body["text_en"] is None


def test_get_article_in_other_law_and_missing(client):
    assert client.get("/articles/2", params={"law": "min-wage-decision-2020"}).json()["number"] == 2
    assert client.get("/articles/999").status_code == 404
    assert client.get("/articles/1", params={"law": "nope"}).status_code == 400


def test_list_articles(client):
    body = client.get("/articles").json()
    assert [a["label"] for a in body] == ["49", "54", "73", "79"]


def test_static_ui_is_served_when_built(engine, tmp_path):
    (tmp_path / "index.html").write_text("<!doctype html><title>qanun</title>", encoding="utf-8")
    app = create_app(Assistant(engine), static_dir=tmp_path)
    with TestClient(app) as c:
        assert "qanun" in c.get("/").text
        assert c.get("/health").json()["status"] == "ok"
