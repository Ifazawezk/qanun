import pytest

from qanun.config import CORPUS_DIR
from qanun.corpus.store import load_articles
from qanun.evaluation import (
    MethodResult,
    Question,
    check_gold,
    load_questions,
    recall_at_k,
    reciprocal_rank,
    render_markdown,
    run_method,
    subsets,
)
from qanun.retrieval.base import Hit


def test_recall_at_k():
    assert recall_at_k(["a", "b", "c"], ["b"], 1) == 0
    assert recall_at_k(["a", "b", "c"], ["b"], 2) == 1
    assert recall_at_k(["a", "b", "c"], ["a", "z"], 3) == 0.5


def test_reciprocal_rank():
    assert reciprocal_rank(["a", "b", "c"], ["c"]) == pytest.approx(1 / 3)
    assert reciprocal_rank(["a", "b"], ["z"]) == 0
    assert reciprocal_rank(["a", "b", "c"], ["c"], cutoff=2) == 0


class StubRetriever:
    name = "stub"

    def __init__(self, answers):
        self.answers = answers

    def search(self, query, k=10, laws=None):
        return [Hit(doc, 1.0, i + 1) for i, doc in enumerate(self.answers[query][:k])]


def test_run_method_and_metrics(sample_articles):
    questions = [
        Question("q1", "en", "t", "first", ("labour-2004:54",)),
        Question("q2", "ar", "t", "second", ("labour-2004:79",)),
    ]
    stub = StubRetriever({
        "first": ["labour-2004:54", "labour-2004:73"],
        "second": ["labour-2004:73", "labour-2004:79"],
    })
    result = run_method("stub", stub, questions)
    metrics = result.metrics(questions)
    assert metrics["recall@1"] == 0.5
    assert metrics["recall@3"] == 1.0
    assert metrics["mrr@10"] == pytest.approx((1 + 0.5) / 2)

    articles = {a.id: a for a in sample_articles}
    groups = subsets(questions, articles)
    assert [q.id for q in groups["en"]] == ["q1"]
    # q1's gold has English text, so it is not in the cross-lingual slice.
    assert groups["en, gold only in Arabic"] == []

    report = render_markdown([result], questions, articles, {"date": "2026-01-01", "articles": "6"})
    assert "| stub | 0.500 |" in report
    assert "## Misses for stub" in report
    assert report.rstrip().endswith("None.")


def test_check_gold_rejects_unknown_articles(sample_articles):
    articles = {a.id: a for a in sample_articles}
    with pytest.raises(ValueError, match="not in the corpus"):
        check_gold([Question("q", "en", "t", "x", ("labour-2004:999",))], articles)
    with pytest.raises(ValueError, match="not in force"):
        check_gold([Question("q", "en", "t", "x", ("residency-2015:22",))], articles)


@pytest.mark.skipif(not (CORPUS_DIR / "articles.jsonl").exists(), reason="corpus not built")
def test_question_set_is_valid():
    questions = load_questions()
    assert 40 <= len(questions) <= 60
    assert len({q.id for q in questions}) == len(questions)
    assert {q.lang for q in questions} == {"en", "ar"}
    check_gold(questions, {a.id: a for a in load_articles()})


def test_method_result_handles_empty_subset():
    assert MethodResult("x").metrics([]) == {}
