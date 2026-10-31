import subprocess
import sys

import numpy as np

from qanun.app_factory import reranker_model
from qanun.retrieval.base import Hit
from qanun.retrieval.dense import DenseIndex
from qanun.retrieval.rerank import Reranker
from qanun.search import SearchEngine


# ---------------------------------------------------------------- reranker


class FakeScorer:
    name = "fake-scorer"

    def __init__(self, favourite):
        self.favourite = favourite
        self.pairs = []

    def score(self, pairs):
        self.pairs += list(pairs)
        return np.array([1.0 if self.favourite in text else 0.0 for _, text in pairs])


def hits_for(*ids):
    return [Hit(doc, 1.0, rank) for rank, doc in enumerate(ids, start=1)]


def test_reranker_reorders_by_the_best_passage(sample_articles, sample_laws):
    scorer = FakeScorer("Ramadan")
    reranker = Reranker(scorer, sample_articles, sample_laws)
    ranked = reranker.rerank("hours", hits_for("labour-2004:54", "labour-2004:73"), k=2)
    assert [h.article_id for h in ranked] == ["labour-2004:73", "labour-2004:54"]
    assert [h.rank for h in ranked] == [1, 2]


def test_reranker_keeps_hits_below_its_depth(sample_articles, sample_laws):
    reranker = Reranker(FakeScorer("Ramadan"), sample_articles, sample_laws, depth=1)
    ids = ("labour-2004:54", "labour-2004:73", "labour-2004:79")
    assert [h.article_id for h in reranker.rerank("hours", hits_for(*ids), k=3)] == list(ids)


def test_reranker_pairs_each_passage_with_the_question_in_its_language(sample_articles, sample_laws):
    scorer = FakeScorer("nothing")
    Reranker(scorer, sample_articles, sample_laws).rerank(
        "working hours", hits_for("labour-2004:73"), 1, translation="ساعات العمل"
    )
    queries = {query for query, _ in scorer.pairs}
    assert queries == {"working hours", "ساعات العمل"}
    for query, passage in scorer.pairs:
        assert ("Ramadan" in passage) == (query == "working hours")


def test_engine_reranks_only_when_a_reranker_is_set(sample_articles, sample_laws, fake_encoder):
    dense = DenseIndex.build(sample_articles, fake_encoder, sample_laws)
    plain = SearchEngine(sample_articles, sample_laws.values(), dense)
    first, second = (h.article_id for h in plain.search("annual leave gratuity", k=2))
    favourite = plain.article(second).text_ar[:20]
    reranker = Reranker(FakeScorer(favourite), sample_articles, sample_laws)
    engine = SearchEngine(sample_articles, sample_laws.values(), dense, reranker)
    assert engine.search("annual leave gratuity", k=1)[0].article_id == second
    assert plain.search("annual leave gratuity", k=1)[0].article_id == first


def test_reranker_module_does_not_import_pytorch():
    code = (
        "import sys; import qanun.retrieval.rerank; "
        "sys.exit(1 if 'torch' in sys.modules or 'sentence_transformers' in sys.modules else 0)"
    )
    assert subprocess.run([sys.executable, "-c", code], capture_output=True).returncode == 0


# ---------------------------------------------------------------- when it is on


def test_reranker_defaults_follow_the_measurements():
    assert reranker_model("e5-small", translating=True, settings={}) == "mmarco"
    assert reranker_model("e5-small", translating=False, settings={}) is None


def test_desktop_build_never_loads_a_reranker():
    assert reranker_model(None, translating=True, settings={}) is None


def test_reranker_can_be_set_or_turned_off():
    assert reranker_model("e5-small", True, {"QANUN_RERANKER": "off"}) is None
    assert reranker_model("e5-small", False, {"QANUN_RERANKER": "some/model"}) == "some/model"


def test_a_named_method_is_not_reranked(sample_articles, sample_laws, fake_encoder):
    dense = DenseIndex.build(sample_articles, fake_encoder, sample_laws)
    scorer = FakeScorer("nothing")
    engine = SearchEngine(sample_articles, sample_laws.values(), dense, Reranker(scorer, sample_articles, sample_laws))
    engine.search("gratuity", method="hybrid")
    engine.search("gratuity", rerank=False)
    assert scorer.pairs == []
    engine.search("gratuity")
    assert scorer.pairs


def test_a_reranker_that_cannot_load_leaves_search_working(monkeypatch):
    def fail(name):
        raise OSError("model is not available")

    monkeypatch.setattr("qanun.search.CrossEncoderScorer", fail)
    engine = SearchEngine.load(None, "mmarco")
    assert engine.reranker is None
    assert engine.search("gratuity", k=1)
