"""Runs data/corpus/checks.yaml with qatar-data-checks, which the dev extra installs."""

import pytest

from qanun.config import CORPUS_DIR

qdc = pytest.importorskip("qatar_data_checks")

ARTICLES = CORPUS_DIR / "articles.jsonl"
pytestmark = pytest.mark.skipif(not ARTICLES.exists(), reason="corpus not built")


@pytest.fixture(scope="module")
def report():
    suite = qdc.Suite.from_yaml(CORPUS_DIR / "checks.yaml")
    return suite.run(qdc.read_table(ARTICLES), base_dir=CORPUS_DIR)


def test_corpus_passes_its_checks(report):
    failed = [f"{r.column} {r.check}: {r.message}" for r in report.results if r.status == "fail"]
    assert failed == []


def test_stored_arabic_has_only_problems_that_search_normalises(report):
    """The text stays as published, so these are warnings. canonical() folds Unicode order
    and Persian letters before indexing. Any other kind of problem needs a look first."""
    found = set()
    for r in report.results:
        if r.check == "arabic_text" and r.status == "warn":
            found |= set(r.details)
    assert found <= {"not_nfc", "persian_letters"}
