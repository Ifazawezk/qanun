import numpy as np
import pytest

from qanun.retrieval.dense import DenseIndex, index_dir_for, load_or_build, resolve_model


def test_build_skips_repealed_and_covers_both_languages(sample_articles, sample_laws, fake_encoder):
    index = DenseIndex.build(sample_articles, fake_encoder, sample_laws)
    assert "residency-2015:22" not in index.doc_ids
    langs = {(p.article_id, p.lang) for p in index.passages}
    assert ("labour-2004:73", "en") in langs
    assert ("labour-2004:73", "ar") in langs
    assert ("labour-2004:79", "en") not in langs  # no English text for this one
    assert np.allclose(np.linalg.norm(index.embeddings, axis=1), 1.0)


def test_english_query_reaches_arabic_only_article(sample_articles, sample_laws, fake_encoder):
    index = DenseIndex.build(sample_articles, fake_encoder, sample_laws)
    hits = index.search("annual leave", k=3)
    assert hits[0].article_id == "labour-2004:79"


def test_article_score_is_best_passage(sample_articles, fake_encoder):
    index = DenseIndex.build(sample_articles, fake_encoder)
    per_passage = index.passage_scores("working hours in Ramadan")
    per_article = index.scores("working hours in Ramadan")
    i = index.doc_ids.index("labour-2004:73")
    rows = [n for n, p in enumerate(index.passages) if p.article_id == "labour-2004:73"]
    assert per_article[i] == pytest.approx(per_passage[rows].max())


def test_law_filter(sample_articles, fake_encoder):
    index = DenseIndex.build(sample_articles, fake_encoder)
    hits = index.search("الحد الأدنى للأجر", laws=["labour-2004"])
    assert all(h.article_id.startswith("labour-2004:") for h in hits)


def test_save_and_load_round_trip(tmp_path, sample_articles, sample_laws, fake_encoder):
    index = DenseIndex.build(sample_articles, fake_encoder, sample_laws)
    index.save(tmp_path)
    loaded = DenseIndex.load(tmp_path, fake_encoder, sample_articles, sample_laws)
    assert np.array_equal(loaded.embeddings, index.embeddings)
    assert loaded.search("gratuity")[0].article_id == index.search("gratuity")[0].article_id


def test_load_rejects_stale_corpus(tmp_path, sample_articles, fake_encoder):
    DenseIndex.build(sample_articles, fake_encoder).save(tmp_path)
    changed = sample_articles[:-2]
    with pytest.raises(ValueError, match="out of date"):
        DenseIndex.load(tmp_path, fake_encoder, changed)


def test_load_or_build_rebuilds_when_stale(tmp_path, sample_articles, fake_encoder):
    load_or_build("fake", sample_articles, root=tmp_path, encoder=fake_encoder)
    assert (index_dir_for("fake", tmp_path) / "embeddings.npy").exists()
    index = load_or_build("fake", sample_articles[:3], root=tmp_path, encoder=fake_encoder)
    assert len(index.doc_ids) == 3


def test_resolve_model_presets():
    assert resolve_model("e5-small").query_prefix == "query: "
    assert resolve_model("some/other-model").model_name == "some/other-model"
