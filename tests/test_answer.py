import pytest

from qanun.answer.assistant import Assistant
from qanun.answer.highlight import best_passage, render_marked, split_sentences
from qanun.retrieval.dense import DenseIndex
from qanun.search import SearchEngine

ARTICLE_79 = (
    "يستحق العامل الذي أمضى في خدمة صاحب العمل سنة كاملة مستمرة، إجازة سنوية بالأجر. "
    "ويجب ألا تقل هذه الإجازة عن ثلاثة أسابيع للعامل الذي تقل خدمته عن خمس سنوات.\n"
    "ويستحق العامل إجازة عن كسور السنة، بنسبة المدة التي قضاها في العمل."
)


def test_split_sentences_merges_short_fragments():
    text = "The Employer may dismiss the Worker in the following cases:\n1.\nIf the Worker lies.\n2. If he is absent for seven days."
    spans = [text[a:b] for a, b in split_sentences(text)]
    assert spans[0] == "The Employer may dismiss the Worker in the following cases:"
    assert spans[1].startswith("1.") and "lies" in spans[1]


def test_best_passage_picks_matching_sentence_and_highlights_terms():
    passage = best_passage("كسور السنة في الإجازة", ARTICLE_79, "ar")
    assert passage.text.startswith("ويستحق العامل إجازة عن كسور السنة")
    marked = {passage.text[a:b] for a, b in passage.highlights}
    assert {"كسور", "السنة", "إجازة"} <= marked


def test_best_passage_uses_similarity_when_no_words_overlap():
    def similarity(sentences):
        return [1.0 if "ثلاثة أسابيع" in s else 0.0 for s in sentences]

    passage = best_passage("How long is annual leave?", ARTICLE_79, "ar", similarity)
    assert "ثلاثة أسابيع" in passage.text
    assert passage.highlights == []


def test_lead_in_sentence_pulls_in_the_list_that_follows():
    text = "The Employer may dismiss the Worker without notice in the following cases:\n1. If the Worker discloses secrets of the establishment."
    passage = best_passage("dismiss without notice", text, "en")
    assert "discloses secrets" in passage.text


def test_common_words_are_not_highlighted_with_idf():
    idf = {"worker": 0.2, "annual": 2.5, "leav": 2.0}
    passage = best_passage("annual leave for a worker", "The Worker is entitled to annual leave.", "en", idf=idf)
    marked = [passage.text[a:b] for a, b in passage.highlights]
    assert marked == ["annual", "leave"]


def test_render_marked():
    passage = best_passage("annual leave", "The Worker is entitled to annual leave.", "en")
    assert render_marked(passage) == "The Worker is entitled to **annual** **leave**."


@pytest.fixture()
def assistant(sample_articles, sample_laws, fake_encoder):
    dense = DenseIndex.build(sample_articles, fake_encoder, sample_laws)
    return Assistant(SearchEngine(sample_articles, sample_laws.values(), dense), top_k=3)


def test_extractive_answer_in_english(assistant):
    answer = assistant.ask("How many hours can I work in Ramadan?")
    assert answer.mode == "extractive"
    assert answer.language == "en"
    assert not answer.refused
    top = answer.citations[0]
    assert top.article_id == "labour-2004:73"
    assert top.title == "Article 73"
    assert top.law_name == "Labour Law"
    assert top.passage.lang == "en"
    assert "Ramadan" in {top.passage.text[a:b] for a, b in top.passage.highlights}


def test_extractive_answer_in_arabic_uses_arabic_text(assistant):
    answer = assistant.ask("ما هي مدة الإجازة السنوية؟")
    assert answer.language == "ar"
    top = answer.citations[0]
    assert top.article_id == "labour-2004:79"
    assert top.title == "المادة 79"
    assert top.passage.lang == "ar"


def test_english_question_on_arabic_only_article_shows_arabic_passage(assistant):
    answer = assistant.ask("annual leave")
    top = answer.citations[0]
    assert top.article_id == "labour-2004:79"
    assert top.passage.lang == "ar"


def test_generative_mode_without_llm_is_an_error(assistant):
    with pytest.raises(RuntimeError):
        assistant.ask("annual leave", mode="generative")
    with pytest.raises(ValueError):
        assistant.ask("annual leave", mode="poetry")


def test_no_results(sample_articles, sample_laws):
    lexical = Assistant(SearchEngine(sample_articles, sample_laws.values()))
    answer = lexical.ask("zzzz qqqq")
    assert answer.refused
    assert answer.citations == []
