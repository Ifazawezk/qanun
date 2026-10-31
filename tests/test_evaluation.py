import pytest

from qanun.config import CORPUS_DIR
from qanun.corpus.store import load_articles
from qanun.evaluation import (
    MethodResult,
    Question,
    RerankedRetriever,
    TranslatedRetriever,
    check_gold,
    load_questions,
    make_translations,
    paired_bootstrap,
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
    assert "Recall@4" in report
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


def test_recall_at_4_is_reported():
    questions = [Question("q", "en", "t", "x", ("d",))]
    result = MethodResult("m", {"q": ["a", "b", "c", "d"]})
    metrics = result.metrics(questions)
    assert metrics["recall@3"] == 0
    assert metrics["recall@4"] == 1


def test_paired_bootstrap_on_a_real_difference_excludes_zero():
    mean, low, high = paired_bootstrap([0.0] * 40, [1.0] * 30 + [0.0] * 10)
    assert mean == pytest.approx(0.75)
    assert 0 < low < mean < high


def test_paired_bootstrap_on_noise_spans_zero():
    mean, low, high = paired_bootstrap([1.0, 0.0] * 20, [0.0, 1.0] * 20)
    assert mean == 0
    assert low < 0 < high


def test_paired_bootstrap_is_repeatable():
    a, b = [0.0, 1.0, 1.0, 0.0], [1.0, 1.0, 0.0, 0.0]
    assert paired_bootstrap(a, b) == paired_bootstrap(a, b)


def test_report_compares_methods_with_the_baseline(sample_articles):
    questions = [Question("q1", "en", "t", "x", ("labour-2004:54",))]
    old = MethodResult("old", {"q1": ["labour-2004:73"]})
    new = MethodResult("new", {"q1": ["labour-2004:54"]})
    articles = {a.id: a for a in sample_articles}
    meta = {"date": "2026-01-01", "articles": "6"}
    report = render_markdown([old, new], questions, articles, meta, baseline="old")
    assert "## Change from old" in report
    assert "| new | +1.000" in report
    assert "Change from" not in render_markdown([old, new], questions, articles, meta)


class StubTranslator:
    def __init__(self, stop_after=None):
        self.calls = []
        self.stop_after = stop_after

    def translate(self, question, lang):
        if self.stop_after is not None and len(self.calls) >= self.stop_after:
            raise RuntimeError("limit reached")
        self.calls.append(question)
        return f"t:{question}"


def test_translations_are_saved_and_reused(tmp_path):
    path = tmp_path / "t.json"
    questions = [Question("q1", "en", "t", "one", ("a",)), Question("q2", "ar", "t", "two", ("a",))]
    first = StubTranslator()
    assert make_translations(questions, first, path, "m") == {"one": "t:one", "two": "t:two"}
    second = StubTranslator()
    assert make_translations(questions, second, path, "m") == {"one": "t:one", "two": "t:two"}
    assert second.calls == []


def test_translations_survive_a_stop_part_way(tmp_path):
    path = tmp_path / "t.json"
    questions = [Question("q1", "en", "t", "one", ("a",)), Question("q2", "ar", "t", "two", ("a",))]
    with pytest.raises(RuntimeError):
        make_translations(questions, StubTranslator(stop_after=1), path, "m")
    resumed = StubTranslator()
    assert make_translations(questions, resumed, path, "m") == {"one": "t:one", "two": "t:two"}
    assert resumed.calls == ["two"]


def test_missing_translations_without_a_key_is_an_error_not_a_guess(tmp_path):
    with pytest.raises(RuntimeError, match="no LLM key"):
        make_translations([Question("q", "en", "t", "x", ("a",))], None, tmp_path / "t.json")


def test_translated_and_reranked_retrievers_pass_the_translation_on():
    class Cross:
        name = "cross"

        def search(self, query, k=10, laws=None, translation=None):
            return [Hit(translation or "none", 1.0, 1)]

    class Ranker:
        depth = 20

        def rerank(self, query, hits, k, translation):
            return [Hit(f"{hits[0].article_id}|{translation}", 1.0, 1)]

    base = TranslatedRetriever(Cross(), {"q": "t"})
    assert base.search("q")[0].article_id == "t"
    assert base.search("other")[0].article_id == "none"
    assert RerankedRetriever(base, Ranker(), {"q": "t"}).search("q")[0].article_id == "t|t"
