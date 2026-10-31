import pytest

from qanun.answer.numbers import numbers_in


@pytest.mark.parametrize(
    "text, expected",
    [
        ("36 hours", {36}),
        ("٣٦ ساعة", {36}),
        ("(1,000) ريال", {1000}),
        ("25%", {25}),
        ("thirty-six hours", {36}),
        ("forty eight hours", {48}),
        ("two hundred and fifty riyals", {250}),
        ("ten thousand riyals", {10000}),
        ("ست وثلاثين ساعة", {36}),
        ("ستة وثلاثون ساعة", {36}),
        ("ثلاثة وستين ساعة", {63}),
        ("ثمان وأربعين ساعة", {48}),
        ("خمسة عشر يوماً", {15}),
        ("اثني عشر أسبوعاً", {12}),
        ("خمسون يوماً", {50}),
        ("ثلاثمائة ريال", {300}),
        ("ألف ريال", {1000}),
        ("ألفي ريال", {2000}),
        ("عشرة آلاف ريال", {10000}),
        ("ثلاثة آلاف ريال", {3000}),
        ("ثلاثة أسابيع ثم أربعة أسابيع", {3, 4}),
    ],
)
def test_numbers_in(text, expected):
    assert numbers_in(text) == {float(n) for n in expected}


@pytest.mark.parametrize("text", ["one week", "2 days", "أسبوعين", "يوم واحد", "the worker", "صاحب العمل"])
def test_one_two_and_plain_words_are_ignored(text):
    assert numbers_in(text) == set()


def test_words_for_different_values_stay_different():
    assert numbers_in("ست وثلاثين") != numbers_in("ثلاثة وستين")
