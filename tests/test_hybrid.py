import pytest

from qanun.retrieval.base import Hit
from qanun.retrieval.bm25 import BM25Index
from qanun.retrieval.dense import DenseIndex
from qanun.retrieval.hybrid import HybridRetriever, reciprocal_rank_fusion


def ranking(*ids):
    return [Hit(doc, 1.0 / rank, rank) for rank, doc in enumerate(ids, start=1)]


def test_rrf_scores_match_formula():
    fused = reciprocal_rank_fusion([ranking("a", "b"), ranking("b", "c")], k=60)
    scores = {h.article_id: h.score for h in fused}
    assert scores["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert scores["a"] == pytest.approx(1 / 61)
    assert scores["c"] == pytest.approx(1 / 62)
    assert [h.article_id for h in fused] == ["b", "a", "c"]
    assert [h.rank for h in fused] == [1, 2, 3]


def test_rrf_document_found_by_one_retriever_still_ranks():
    fused = reciprocal_rank_fusion([ranking("a"), []])
    assert [h.article_id for h in fused] == ["a"]


def test_rrf_weights():
    fused = reciprocal_rank_fusion([ranking("a"), ranking("b")], weights=[1.0, 2.0])
    assert fused[0].article_id == "b"
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([ranking("a")], weights=[1.0, 2.0])


def test_hybrid_combines_lexical_and_cross_lingual_hits(sample_articles, sample_laws, fake_encoder):
    bm25 = BM25Index.from_articles(sample_articles, sample_laws)
    dense = DenseIndex.build(sample_articles, fake_encoder, sample_laws)
    hybrid = HybridRetriever([bm25, dense])

    # Article 79 has no English text: BM25 cannot see it for an English query,
    # the dense side can, and the fused list keeps it.
    assert "labour-2004:79" not in [h.article_id for h in bm25.search("annual leave")]
    hits = hybrid.search("annual leave", k=3)
    assert hits[0].article_id == "labour-2004:79"

    rankings = hybrid.rankings("annual leave")
    assert set(rankings) == {"bm25", "dense"}


def test_hybrid_keeps_rankings_from_retrievers_with_the_same_name():
    class Fixed:
        name = "dense"

        def __init__(self, ids):
            self.ids = ids

        def search(self, query, k=10, laws=None):
            return ranking(*self.ids)

    hybrid = HybridRetriever([Fixed(["a", "b"]), Fixed(["c", "a"])])
    assert set(hybrid.rankings("q")) == {"dense", "dense#2"}
    assert [h.article_id for h in hybrid.search("q")] == ["a", "c", "b"]


def test_hybrid_law_filter(sample_articles, sample_laws, fake_encoder):
    bm25 = BM25Index.from_articles(sample_articles, sample_laws)
    dense = DenseIndex.build(sample_articles, fake_encoder, sample_laws)
    hits = HybridRetriever([bm25, dense]).search("الأجر", laws=["min-wage-decision-2020"])
    assert {h.article_id for h in hits} == {"min-wage-decision-2020:2"}


def test_search_engine_methods(sample_articles, sample_laws, fake_encoder):
    from qanun.search import SearchEngine

    dense = DenseIndex.build(sample_articles, fake_encoder, sample_laws)
    engine = SearchEngine(sample_articles, sample_laws.values(), dense)
    assert engine.methods == ["bm25", "dense", "hybrid"]
    assert engine.default_method == "hybrid"
    assert engine.search("gratuity", k=1)[0].article_id == "labour-2004:54"
    assert engine.find("labour-2004", "54").number == 54

    lexical_only = SearchEngine(sample_articles, sample_laws.values())
    assert lexical_only.methods == ["bm25"]
    with pytest.raises(ValueError):
        lexical_only.search("gratuity", method="dense")
