"""qatar-abtest computes paired bootstrap intervals with its own code. On the published
rankings in eval/results.json it must give the same intervals as qanun's evaluation,
for every pair of methods. The dev extra installs it."""

import json
from itertools import combinations

import pytest

from qanun.evaluation import EVAL_DIR, MethodResult, load_questions, paired_bootstrap, question_scores

compare = pytest.importorskip("qatar_abtest.compare")


def test_qabtest_gives_the_same_intervals():
    questions = load_questions()
    saved = json.loads((EVAL_DIR / "results.json").read_text(encoding="utf-8"))
    methods = [MethodResult(m["method"], m["rankings"]) for m in saved["methods"]]
    for old, new in combinations(methods, 2):
        for metric in ("recall@4", "recall@5", "mrr@10"):
            a, b = question_scores(old, questions, metric), question_scores(new, questions, metric)
            assert compare.paired_bootstrap(a, b) == paired_bootstrap(a, b), (old.method, new.method, metric)
