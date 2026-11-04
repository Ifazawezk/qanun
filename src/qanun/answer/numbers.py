"""Reads numbers from legal text so that an answer's figures can be checked.

Law texts mix digits with words, and models get the words wrong. One real answer
turned the 36 hours of Article 73 (ست وثلاثين) into 63 (ثلاثة وستين), which every
citation check passes. Comparing by value catches that, and also accepts a correct
answer that writes the same number in another form, such as 36 for ست وثلاثين.

One and two are ignored: Arabic has a dual form (أسبوعين is two weeks) that holds
no number word, so they cannot be compared fairly.
"""

from __future__ import annotations

import re
from itertools import combinations

from qanun.text.normalize import DIGITS, normalize_arabic

SMALLEST = 3
DIGIT_NUMBER = re.compile(r"\d+(?:[,،]\d{3})*(?:\.\d+)?")
ARABIC_WORD = re.compile(r"[ء-ي]+")
ENGLISH_WORD = re.compile(r"[A-Za-z]+")
PREFIX = re.compile(r"^(?:و|ب|ل|ف|ك)?(?:ال)?")

ENGLISH = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40,
    "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}
ARABIC_SPELLINGS = {
    3: "ثلاث ثلاثة", 4: "اربع اربعة", 5: "خمس خمسة", 6: "ست ستة", 7: "سبع سبعة",
    8: "ثمان ثماني ثمانية", 9: "تسع تسعة", 10: "عشر عشرة",
    20: "عشرون عشرين", 30: "ثلاثون ثلاثين", 40: "اربعون اربعين", 50: "خمسون خمسين",
    60: "ستون ستين", 70: "سبعون سبعين", 80: "ثمانون ثمانين", 90: "تسعون تسعين",
    100: "مئة مائة", 200: "مئتان مئتين مائتان مائتين", 300: "ثلاثمئة ثلاثمائة",
    400: "اربعمئة اربعمائة", 500: "خمسمئة خمسمائة", 600: "ستمئة ستمائة", 700: "سبعمئة سبعمائة",
    800: "ثمانمئة ثمانمائة", 900: "تسعمئة تسعمائة", 1000: "الف", 2000: "الفان الفين الفي",
}


def _fold(word: str) -> str:
    return normalize_arabic(word)


ARABIC = {_fold(word): value for value, spellings in ARABIC_SPELLINGS.items() for word in spellings.split()}
ARABIC_THOUSANDS = {_fold("آلاف"), _fold("ألفا"), _fold("ألفاً")}
ARABIC_TEN_PLUS = {_fold("اثني"), _fold("اثنا"), _fold("اثنتي"), _fold("اثنتا")}


def _arabic_value(word: str) -> int | str | None:
    folded = _fold(word)
    for candidate in (folded, PREFIX.sub("", folded, count=1)):
        if candidate in ARABIC:
            return ARABIC[candidate]
        if candidate in ARABIC_THOUSANDS:
            return "thousands"
        if candidate in ARABIC_TEN_PLUS:
            return 2  # the 2 of "twelve", kept only because عشر follows it
    return None


def _arabic_numbers(text: str) -> set[float]:
    found: set[float] = set()
    total = 0
    for word in ARABIC_WORD.findall(text):
        value = _arabic_value(word)
        if value == "thousands":
            total = max(total, 1) * 1000
        elif isinstance(value, int):
            total += value
        else:
            if total:
                found.add(float(total))
            total = 0
    if total:
        found.add(float(total))
    return found


def _english_numbers(text: str) -> set[float]:
    found: set[float] = set()
    total = current = 0
    for word in ENGLISH_WORD.findall(text.lower()):
        if word in ENGLISH:
            current += ENGLISH[word]
        elif word == "hundred":
            current = max(current, 1) * 100
        elif word == "thousand":
            total += max(current, 1) * 1000
            current = 0
        elif word == "and" and (total or current):
            continue
        else:
            if total + current:
                found.add(float(total + current))
            total = current = 0
    if total + current:
        found.add(float(total + current))
    return found


def numbers_in(text: str) -> set[float]:
    """Every number of 3 or more in the text, written in digits or in words."""
    plain = text.translate(DIGITS)
    found = {float(m.replace(",", "").replace("،", "")) for m in DIGIT_NUMBER.findall(plain)}
    found |= _arabic_numbers(plain) | _english_numbers(plain)
    return {n for n in found if n >= SMALLEST}


# ---------------------------------------------------------------- simple arithmetic
#
# A correct answer may add up or convert figures that the article states: 3 days for
# each Eid is 6 days, and four weeks is 28 days. Only two figures are ever summed,
# and only figures with the same unit in the same article. Three would be too loose:
# Article 90 has 36, 24 and 3 hours, which add up to the 63 of a real wrong answer.

UNITS = {
    "day": "يوم يوما ايام day days",
    "week": "اسبوع اسبوعا اسابيع week weeks",
    "month": "شهر شهرا اشهر شهور month months",
    "hour": "ساعه ساعات hour hours",
}
UNIT_OF = {word: unit for unit, words in UNITS.items() for word in words.split()}
DUAL = {"يومين": "day", "يومان": "day", "اسبوعين": "week", "اسبوعان": "week", "شهرين": "month",
        "شهران": "month", "ساعتين": "hour", "ساعتان": "hour"}
# Words that may sit between a number and its unit, as in "three working days".
BETWEEN = {"working", "work", "consecutive", "calendar", "paid", "full"}
MONTH_WORDS = {"شهر", "الشهر", "month", "the"}
DAYS_IN_WEEK = 7
TOKEN = re.compile(r"\d+(?:,\d{3})*(?:\.\d+)?|[ء-ي]+|[A-Za-z]+")
LIST_MARKER = re.compile(r"^\s*\d+\s*[-.)]")


def _word_number(word: str) -> bool:
    if word in ENGLISH or word in ("hundred", "thousand"):
        return True
    return _arabic_value(word) is not None


def _value_before(words: list[str], end: int) -> float | None:
    """The number written just before position end, in digits or in words."""
    if end <= 0:
        return None
    if words[end - 1][0].isdigit():
        return float(words[end - 1].replace(",", ""))
    start = end
    while start > 0 and (_word_number(words[start - 1]) or (words[start - 1] == "and" and start < end)):
        start -= 1
    phrase = " ".join(words[start:end])
    found = _english_numbers(phrase) | _arabic_numbers(phrase)
    found |= {float(ENGLISH[w]) for w in words[start:end] if w in ("one", "two")}
    return max(found) if found else None


def _figures(text: str) -> list[tuple[float, str, bool]]:
    """(value, unit, per day) for each figure with a unit, such as 8 hours a day."""
    figures = []
    for line in normalize_arabic(text.translate(DIGITS)).lower().split("\n"):
        words = TOKEN.findall(LIST_MARKER.sub("", line))
        for i, word in enumerate(words):
            unit = UNIT_OF.get(word)
            if word in DUAL:
                value, unit = 2.0, DUAL[word]
            elif unit is None:
                continue
            else:
                end = i - 1 if i and words[i - 1] in BETWEEN else i
                value = _value_before(words, end)
                if value is None:
                    continue
            after = " ".join(words[i + 1 : i + 3])
            per_day = after.startswith(("يوميا", "في اليوم", "per day", "a day", "each day", "daily"))
            figures.append((value, unit, per_day))
    return figures


def _days_in_month(text: str) -> float | None:
    """A month counts as days only where the article says how many."""
    words = TOKEN.findall(normalize_arabic(text.translate(DIGITS)).lower())
    for i, word in enumerate(words):
        if word in ("شهر", "الشهر", "month"):
            for unit_at in range(i + 2, min(i + 6, len(words))):
                if UNIT_OF.get(words[unit_at]) == "day":
                    value = _value_before(words, unit_at)
                    if value is not None and 28 <= value <= 31:
                        return value
    return None


def derived_numbers(text: str) -> set[float]:
    """Figures a correct answer can reach with simple arithmetic on one text: the sum
    of two figures with the same unit, weeks as days, months as days where the text
    says how long a month is, and hours a day times a number of days."""
    figures = _figures(text)
    found: set[float] = set()
    for (a, unit_a, _), (b, unit_b, _) in combinations(figures, 2):
        if unit_a == unit_b:
            found.add(a + b)
    month = _days_in_month(text)
    days = [value for value, unit, _ in figures if unit == "day"]
    for value, unit, per_day in figures:
        if unit == "week":
            found.add(value * DAYS_IN_WEEK)
        elif unit == "month" and month:
            found.add(value * month)
        elif unit == "hour" and per_day:
            found.update(value * d for d in days)
    return {n for n in found if n >= SMALLEST}
