import subprocess
import sys

import pytest

from qanun.retrieval.base import Hit
from qanun.retrieval.bm25 import BM25Index
from qanun.retrieval.dense import DenseIndex
from qanun.retrieval.hybrid import CrossLanguageRetriever, max_fusion
from qanun.search import SearchEngine


@pytest.fixture()
def indexes(sample_articles, sample_laws, fake_encoder):
    bm25 = BM25Index.from_articles(sample_articles, sample_laws)
    dense = DenseIndex.build(sample_articles, fake_encoder, sample_laws)
    return bm25, dense


# ---------------------------------------------------------------- fusion


def test_max_fusion_keeps_the_best_score_and_does_not_add():
    both = [Hit("both", 0.4, 1)]
    fused = max_fusion([both, both, [Hit("arabic-only", 0.5, 1)]])
    assert [h.article_id for h in fused] == ["arabic-only", "both"]
    assert fused[1].score == 0.4


def test_dense_search_can_be_limited_to_one_language(indexes):
    _, dense = indexes
    assert "labour-2004:79" not in {h.article_id for h in dense.search("annual leave", lang="en")}
    assert "labour-2004:79" in {h.article_id for h in dense.search("annual leave", lang="ar")}


def test_translation_reaches_an_arabic_only_article_with_keywords_alone(indexes):
    bm25, _ = indexes
    retriever = CrossLanguageRetriever(bm25)
    assert retriever.search("written notice") == []
    hits = retriever.search("written notice", translation="الإخطار كتابة")
    assert hits[0].article_id == "labour-2004:49"


def test_arabic_only_article_is_not_buried_without_a_translation(indexes):
    bm25, dense = indexes
    hits = CrossLanguageRetriever(bm25, dense).search("annual leave", k=3)
    assert "labour-2004:79" in {h.article_id for h in hits}


def test_cross_language_search_respects_the_law_filter(indexes):
    bm25, dense = indexes
    hits = CrossLanguageRetriever(bm25, dense).search(
        "minimum wage", laws=["min-wage-decision-2020"], translation="الحد الأدنى للأجر"
    )
    assert {h.article_id for h in hits} == {"min-wage-decision-2020:2"}


def test_engine_uses_a_translation_even_without_embeddings(sample_articles, sample_laws):
    engine = SearchEngine(sample_articles, sample_laws.values())
    assert engine.search("written notice") == []
    hits = engine.search("written notice", translation="الإخطار كتابة")
    assert hits[0].article_id == "labour-2004:49"


def test_an_explicit_method_ignores_the_translation(sample_articles, sample_laws):
    engine = SearchEngine(sample_articles, sample_laws.values())
    assert engine.search("written notice", method="bm25", translation="الإخطار كتابة") == []


def test_retrieval_modules_do_not_import_pytorch():
    code = (
        "import sys; import qanun.search; "
        "sys.exit(1 if 'torch' in sys.modules or 'sentence_transformers' in sys.modules else 0)"
    )
    assert subprocess.run([sys.executable, "-c", code], capture_output=True).returncode == 0
