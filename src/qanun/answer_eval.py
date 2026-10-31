"""Answer evaluation: does the whole system answer well and refuse when it should?

    qanun eval-answers --sample 20          # small first run
    qanun eval-answers --model allam-2-7b --lang ar --sample 10

Answerable questions come from eval/questions.jsonl. Out-of-scope and trick
questions come from eval/answer_questions.jsonl.

Free tiers allow few tokens a day, so a run is resumable: every reply is cached
in eval/.cache, each run spends a small budget, and stopping at a limit keeps all
progress. Growing --sample keeps the questions already chosen.

Most numbers are computed in code. Faithfulness is not: it comes from an LLM
judge, which can be wrong, and it is labelled that way in the report.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field, replace
from datetime import date
from pathlib import Path
from typing import Iterable, Sequence

from qanun.answer.assistant import Assistant
from qanun.answer.cache import BudgetExhausted, CachedClient
from qanun.answer.generate import (
    CITATION,
    Generator,
    as_data,
    load_prompt,
    split_sentences,
    without_list_marker,
)
from qanun.answer.llm import LLMClient, LLMError
from qanun.evaluation import EVAL_DIR, Question, load_questions

TRICK_PATH = EVAL_DIR / "answer_questions.jsonl"
RESULTS_DIR = EVAL_DIR / "answers"
REPORT_PATH = EVAL_DIR / "answer_results.md"
SHOULD_REFUSE = ("out_of_scope", "unanswerable")
GROUPS = ("answerable", "out_of_scope", "unanswerable", "false_premise", "injection", "planted")


@dataclass(frozen=True)
class Item:
    id: str
    lang: str
    group: str
    question: str
    gold: tuple[str, ...] = ()
    correct_any: tuple[str, ...] = ()
    canary: tuple[str, ...] = ()
    planted: str | None = None


@dataclass
class Record:
    id: str
    lang: str
    group: str
    question: str
    retrieved: list[str]
    gold: list[str]
    mode: str
    refused: bool
    text: str
    cited: list[str]
    removed: list[str]
    dropped: int
    uncited: int
    flagged: list[str] = field(default_factory=list)
    numbers: list[float] = field(default_factory=list)
    error: str | None = None
    premise_corrected: bool = False
    canary_hit: bool = False
    verdicts: list[dict] = field(default_factory=list)
    judge_failed: bool = False


def load_items(path: Path = TRICK_PATH, questions: Sequence[Question] | None = None) -> list[Item]:
    items = [
        Item(q.id, q.lang, "answerable", q.question, q.gold)
        for q in (load_questions() if questions is None else questions)
    ]
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                row = json.loads(line)
                items.append(
                    Item(
                        row["id"], row["lang"], row["kind"], row["question"],
                        correct_any=tuple(row.get("correct_any", ())),
                        canary=tuple(row.get("canary", ())),
                        planted=row.get("planted"),
                    )
                )
    return items


def choose(items: Sequence[Item], sample: int | None, lang: str | None = None) -> list[Item]:
    """A stable sample, half answerable and half not. Order comes from a hash of the
    id, so a larger sample contains every smaller one and reuses its cached replies."""
    pool = [i for i in items if lang is None or i.lang == lang]
    if sample is None or sample >= len(pool):
        return pool

    def order(item: Item) -> str:
        return hashlib.sha256(item.id.encode("utf-8")).hexdigest()

    answerable = sorted((i for i in pool if i.group == "answerable"), key=order)
    others: dict[str, list[Item]] = {}
    for item in sorted((i for i in pool if i.group != "answerable"), key=order):
        others.setdefault(item.group, []).append(item)
    # Take from each kind in turn, so a small sample still covers all of them.
    mixed: list[Item] = []
    while any(others.values()):
        for group in GROUPS:
            if others.get(group):
                mixed.append(others[group].pop(0))
    half = sample // 2
    picked = answerable[: sample - min(half, len(mixed))] + mixed[:half]
    return picked


class PlantingGenerator:
    """Adds an instruction to the first retrieved article, to test that retrieved
    text is treated as data by a real model. Only the evaluation uses this."""

    def __init__(self, generator: Generator, text: str):
        self.generator = generator
        self.text = text

    def answer(self, question, lang, articles, citations):
        first = replace(articles[0], text_ar=f"{articles[0].text_ar}\n{self.text}")
        return self.generator.answer(question, lang, [first, *articles[1:]], citations)


def contains_any(text: str, needles: Sequence[str]) -> bool:
    lowered = text.lower()
    return any(needle.lower() in lowered for needle in needles)


def run_item(item: Item, assistant: Assistant, client: CachedClient) -> Record:
    client.last_error = None
    generator = assistant.generator
    if item.planted:
        generator = PlantingGenerator(assistant.generator, item.planted)
    planted_view = Assistant(assistant.engine, generator, assistant.top_k, assistant.translator)
    answer = planted_view.ask(item.question, mode="generative")
    return Record(
        id=item.id,
        lang=item.lang,
        group=item.group,
        question=item.question,
        retrieved=[c.article_id for c in answer.citations],
        gold=list(item.gold),
        mode=answer.mode,
        refused=answer.refused,
        text=answer.text,
        cited=[c.article_id for c in answer.citations if c.cited],
        removed=answer.removed_citations,
        dropped=answer.dropped_sentences,
        uncited=answer.uncited_sentences,
        flagged=answer.flagged_references,
        numbers=answer.removed_numbers,
        error=client.last_error,
        premise_corrected=not answer.refused and contains_any(answer.text, item.correct_any),
        canary_hit=not answer.refused and contains_any(answer.text, item.canary),
    )


# ---------------------------------------------------------------- LLM judge


def claims_of(text: str) -> list[tuple[str, list[str]]]:
    """Sentences with the ids that cover them. A citation at the end of a line also
    covers the sentences before it, or they would never be checked."""
    claims = []
    for line in text.split("\n"):
        waiting: list[str] = []
        for sentence in split_sentences(line):
            ids = list(dict.fromkeys(CITATION.findall(sentence)))
            plain = re.sub(r"\s+([.,;:،؛])", r"\1", CITATION.sub("", sentence)).strip()
            plain = without_list_marker(plain)
            if plain:
                waiting.append(plain)
            if ids:
                claims += [(claim, ids) for claim in waiting]
                waiting = []
    return claims


def judge_messages(claims: Sequence[tuple[str, list[str]]], articles: dict) -> tuple[str, str]:
    cited = list(dict.fromkeys(i for _, ids in claims for i in ids))
    sources = []
    for article_id in cited:
        article = articles[article_id]
        english = f"\nEnglish (unofficial):\n{as_data(article.text_en)}" if article.text_en else ""
        sources.append(f'<article id="{article_id}">\n{as_data(article.text_ar)}{english}\n</article>')
    numbered = [
        f'<claim n="{n}" cites="{", ".join(ids)}">{as_data(text)}</claim>'
        for n, (text, ids) in enumerate(claims, start=1)
    ]
    return load_prompt("judge"), "\n\n".join(sources) + "\n\n" + "\n".join(numbered)


def parse_verdicts(reply: str, count: int) -> list[dict] | None:
    """The verdicts, or None if the reply cannot be trusted to line up with the claims."""
    start, end = reply.find("{"), reply.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        verdicts = json.loads(reply[start : end + 1])["verdicts"]
        by_claim = {int(v["claim"]): v for v in verdicts}
        if sorted(by_claim) != list(range(1, count + 1)):
            return None
        ordered = [by_claim[n] for n in range(1, count + 1)]
    except (ValueError, KeyError, TypeError):
        return None
    if not all(isinstance(v.get("supported"), bool) for v in ordered):
        return None
    return ordered


def judge(record: Record, articles: dict, client: LLMClient) -> None:
    claims = claims_of(record.text)
    if record.refused or record.mode != "generative" or not claims:
        return
    system, user = judge_messages(claims, articles)
    try:
        verdicts = parse_verdicts(client.complete(system, user), len(claims))
    except LLMError:
        verdicts = None
    if verdicts is None:
        record.judge_failed = True
        return
    record.verdicts = [
        {"claim": text, "supported": v["supported"], "reason": str(v.get("reason", ""))[:200]}
        for (text, _), v in zip(claims, verdicts)
    ]


# ---------------------------------------------------------------- metrics


def _rate(part: int, whole: int) -> float | None:
    return part / whole if whole else None


def citation_escaped(record: Record) -> bool:
    """True if a delivered answer cites something it was not given. The check in
    generate.py should make this impossible, so the report counts it anyway."""
    return record.mode == "generative" and bool(set(CITATION.findall(record.text)) - set(record.retrieved))


def summarize(records: Sequence[Record]) -> dict:
    by_group = {g: [r for r in records if r.group == g] for g in GROUPS}
    answerable = by_group["answerable"]
    usable = [r for r in answerable if r.mode == "generative"]
    answered = [r for r in usable if not r.refused]
    with_gold = [r for r in usable if set(r.gold) & set(r.retrieved)]
    cited = [(r, i) for r in answered for i in r.cited]
    should_refuse = [r for g in SHOULD_REFUSE for r in by_group[g] if r.mode == "generative"]
    premise = [r for r in by_group["false_premise"] if r.mode == "generative"]
    hostile = [r for g in ("injection", "planted") for r in by_group[g] if r.mode == "generative"]
    judged = [r for r in records if r.verdicts]
    verdicts = [v for r in judged for v in r.verdicts]
    return {
        "questions": len(records),
        "generation_failed": sum(r.mode != "generative" for r in records),
        "answerable": {
            "total": len(usable),
            "answered": len(answered),
            "false_refusals": sum(r.refused for r in usable),
            "false_refusal_rate": _rate(sum(r.refused for r in usable), len(usable)),
            "gold_retrieved": len(with_gold),
            "false_refusals_with_gold_retrieved": sum(r.refused for r in with_gold),
            "false_refusal_rate_with_gold_retrieved": _rate(sum(r.refused for r in with_gold), len(with_gold)),
        },
        "citations": {
            "delivered": len(cited),
            "delivered_gold": sum(i in r.gold for r, i in cited),
            "precision_against_gold": _rate(sum(i in r.gold for r, i in cited), len(cited)),
            "answers_citing_a_gold_article": sum(bool(set(r.cited) & set(r.gold)) for r in answered),
            "answers_citing_a_gold_article_rate": _rate(
                sum(bool(set(r.cited) & set(r.gold)) for r in answered), len(answered)
            ),
            "invalid_references_caught": sum(len(r.removed) for r in records),
            "answers_with_invalid_references": sum(bool(r.removed) for r in records),
            "references_flagged_from_source": sum(len(r.flagged) for r in records),
            "sentences_removed": sum(r.dropped for r in records),
            "wrong_numbers_caught": sum(len(r.numbers) for r in records),
            "uncited_sentences": sum(r.uncited for r in records),
            "invalid_citations_reaching_the_user": sum(citation_escaped(r) for r in records),
        },
        "refusal": {
            "should_refuse": len(should_refuse),
            "refused": sum(r.refused for r in should_refuse),
            "refusal_rate": _rate(sum(r.refused for r in should_refuse), len(should_refuse)),
        },
        "false_premise": {
            "total": len(premise),
            "refused": sum(r.refused for r in premise),
            "corrected": sum(r.premise_corrected for r in premise),
        },
        "hostile": {
            "total": len(hostile),
            "canary_in_answer": sum(r.canary_hit for r in hostile),
        },
        "faithfulness_llm_judge": {
            "answers_judged": len(judged),
            "judge_failed": sum(r.judge_failed for r in records),
            "claims": len(verdicts),
            "supported": sum(bool(v["supported"]) for v in verdicts),
            "rate": _rate(sum(bool(v["supported"]) for v in verdicts), len(verdicts)),
        },
    }


def run(
    items: Sequence[Item],
    assistant: Assistant,
    client: CachedClient,
    judge_client: LLMClient | None,
    out_path: Path,
    meta: dict,
) -> dict:
    """Run until done or until the budget or a provider limit stops it. Whatever
    finished is saved, and the next run repeats it from the cache at no cost."""
    articles = assistant.engine.articles
    previous = _saved_ids(out_path)
    # Questions already answered come first: they are cached, so a limit can only stop
    # the run after it has rebuilt everything it had before.
    items = sorted(items, key=lambda item: item.id not in previous)
    records: list[Record] = []
    stopped = None
    try:
        for item in items:
            record = run_item(item, assistant, client)
            if judge_client is not None:
                judge(record, articles, judge_client)
            records.append(record)
    except BudgetExhausted as exc:
        stopped = str(exc)
    payload = {
        "meta": {
            **meta,
            "date": date.today().isoformat(),
            "planned": len(items),
            "finished": len(records),
            "complete": stopped is None,
            "stopped_by": stopped,
            "requests_this_run": client.budget.requests_used,
            "tokens_this_run": client.budget.tokens_used,
            "cache_hits_this_run": client.budget.cache_hits,
        },
        "summary": summarize(records),
        "records": [asdict(r) for r in records],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if stopped and len(records) < len(previous):
        # Never replace fuller results with a run that was cut short.
        out_path = out_path.with_suffix(".partial.json")
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
    return payload


def _saved_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {record["id"] for record in json.loads(path.read_text(encoding="utf-8"))["records"]}


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9.-]+", "-", text.lower()).strip("-")


# ---------------------------------------------------------------- report


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{100 * value:.0f}%"


def render_report(results: Iterable[dict]) -> str:
    results = list(results)
    lines = [
        "# Answer evaluation",
        "",
        "Measured against real models through `qanun eval-answers`. These are small samples, so one "
        "question moves a rate a long way. Every run below states how many questions it covers.",
        "",
        "Citation precision counts a citation as correct only if it is the labelled gold article. Most "
        "questions have one gold article, so an answer that also cites a relevant neighbour lowers it, "
        "which makes it a lower bound.",
        "",
        "## Runs",
        "",
        "| Run | Answer model | Judge | Questions | Finished | Stopped by |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for r in results:
        m = r["meta"]
        lines.append(
            f"| {m['name']} | {m['model']} | {m.get('judge_model') or 'none'} | {m['planned']} "
            f"| {m['finished']} | {m['stopped_by'] or 'finished'} |"
        )
    lines += [
        "",
        "## Measured in code",
        "",
        "| Run | False refusals | False refusals, gold retrieved | Refused when it should | "
        "False premise refused or corrected | Hostile text obeyed | Citation precision | Answers citing gold | "
        "Invalid references caught | Figures caught | Invalid reaching the user |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in results:
        s, name = r["summary"], r["meta"]["name"]
        a, c, f, h, ref = s["answerable"], s["citations"], s["false_premise"], s["hostile"], s["refusal"]
        lines.append(
            f"| {name} | {a['false_refusals']}/{a['total']} ({_pct(a['false_refusal_rate'])}) "
            f"| {a['false_refusals_with_gold_retrieved']}/{a['gold_retrieved']} "
            f"| {ref['refused']}/{ref['should_refuse']} ({_pct(ref['refusal_rate'])}) "
            f"| {f['refused'] + f['corrected']}/{f['total']} | {h['canary_in_answer']}/{h['total']} "
            f"| {c['delivered_gold']}/{c['delivered']} ({_pct(c['precision_against_gold'])}) "
            f"| {c['answers_citing_a_gold_article']}/{a['answered']} "
            f"| {c['invalid_references_caught']} | {c['wrong_numbers_caught']} "
            f"| {c['invalid_citations_reaching_the_user']} |"
        )
    lines += [
        "",
        "## Measured by an LLM judge",
        "",
        "**These numbers come from a language model and can be wrong.** It reads each cited claim next "
        "to the cited articles and says whether they support it. Treat them as an estimate and read "
        "the unsupported claims listed below.",
        "",
        "| Run | Judge | Answers judged | Claims | Supported | Faithfulness | Judge failed |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in results:
        j, m = r["summary"]["faithfulness_llm_judge"], r["meta"]
        lines.append(
            f"| {m['name']} | {m.get('judge_model') or 'none'} | {j['answers_judged']} | {j['claims']} "
            f"| {j['supported']} | {_pct(j['rate'])} | {j['judge_failed']} |"
        )
    for r in results:
        rows = [
            (rec["id"], v["claim"], v["reason"])
            for rec in r["records"] for v in rec["verdicts"] if not v["supported"]
        ]
        if rows:
            lines += ["", f"### Claims the judge rejected: {r['meta']['name']}", "", "| Question | Claim | Reason |", "| --- | --- | --- |"]
            lines += [f"| {i} | {claim} | {reason} |" for i, claim, reason in rows]
    lines.append("")
    return "\n".join(lines)


def write_report(results_dir: Path = RESULTS_DIR, report_path: Path = REPORT_PATH) -> None:
    results = [
        json.loads(path.read_text(encoding="utf-8")) for path in sorted(results_dir.glob("*.json"))
    ]
    report_path.write_text(render_report(results), encoding="utf-8", newline="\n")
