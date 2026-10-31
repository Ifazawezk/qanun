"""Retrieval evaluation: Recall@k and MRR for BM25, dense and hybrid search.

    qanun eval                       # default embedding model
    qanun eval --model e5-small --model minilm
    qanun eval --translate           # also measure query translation
    qanun eval --reranker mmarco     # also measure a reranker

Writes eval/results.json and eval/results.md. Translations are kept in
eval/translations.json, so a key is only needed to make them the first time.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np

from qanun.config import ROOT
from qanun.corpus.models import Article
from qanun.retrieval.base import Hit, Retriever

EVAL_DIR = ROOT / "eval"
QUESTIONS_PATH = EVAL_DIR / "questions.jsonl"
TRANSLATIONS_PATH = EVAL_DIR / "translations.json"
CACHE_PATH = EVAL_DIR / ".cache" / "llm.jsonl"
KS = (1, 3, 4, 5, 10)


@dataclass(frozen=True)
class Question:
    id: str
    lang: str
    topic: str
    question: str
    gold: tuple[str, ...]


def load_questions(path: Path = QUESTIONS_PATH) -> list[Question]:
    questions = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                row = json.loads(line)
                questions.append(
                    Question(row["id"], row["lang"], row["topic"], row["question"], tuple(row["gold"]))
                )
    return questions


def check_gold(questions: Iterable[Question], articles: dict[str, Article]) -> None:
    for q in questions:
        for gold in q.gold:
            if gold not in articles:
                raise ValueError(f"{q.id}: gold article {gold} is not in the corpus")
            if not articles[gold].in_force:
                raise ValueError(f"{q.id}: gold article {gold} is not in force")


def recall_at_k(ranked: Sequence[str], gold: Sequence[str], k: int) -> float:
    return len(set(ranked[:k]) & set(gold)) / len(gold)


def reciprocal_rank(ranked: Sequence[str], gold: Sequence[str], cutoff: int = 10) -> float:
    for rank, doc in enumerate(ranked[:cutoff], start=1):
        if doc in gold:
            return 1.0 / rank
    return 0.0


@dataclass
class MethodResult:
    method: str
    per_question: dict[str, list[str]] = field(default_factory=dict)
    seconds: float = 0.0

    def metrics(self, questions: Sequence[Question]) -> dict[str, float]:
        if not questions:
            return {}
        out = {}
        for k in KS:
            out[f"recall@{k}"] = sum(
                recall_at_k(self.per_question[q.id], q.gold, k) for q in questions
            ) / len(questions)
        out["mrr@10"] = sum(
            reciprocal_rank(self.per_question[q.id], q.gold) for q in questions
        ) / len(questions)
        return out


def run_method(name: str, retriever: Retriever, questions: Sequence[Question], depth: int = 10) -> MethodResult:
    result = MethodResult(name)
    start = time.perf_counter()
    for q in questions:
        result.per_question[q.id] = [h.article_id for h in retriever.search(q.question, k=depth)]
    result.seconds = time.perf_counter() - start
    return result


def subsets(questions: Sequence[Question], articles: dict[str, Article]) -> dict[str, list[Question]]:
    """Overall, by language, and the cross-lingual slice: English questions whose
    gold articles have no English text, so only Arabic text can match."""
    cross = [
        q for q in questions
        if q.lang == "en" and all(not articles[g].text_en for g in q.gold)
    ]
    return {
        "all": list(questions),
        "en": [q for q in questions if q.lang == "en"],
        "ar": [q for q in questions if q.lang == "ar"],
        "en, gold only in Arabic": cross,
    }


class TranslatedRetriever:
    """Cross-language search with translations made in advance."""

    def __init__(self, cross, translations: dict[str, str]):
        self.name = "cross+translation"
        self.cross = cross
        self.translations = translations

    def search(self, query: str, k: int = 10, laws: Iterable[str] | None = None) -> list[Hit]:
        return self.cross.search(query, k=k, laws=laws, translation=self.translations.get(query))


class RerankedRetriever:
    def __init__(self, base: Retriever, reranker, translations: dict[str, str] | None = None):
        self.name = f"{base.name}+rerank"
        self.base = base
        self.reranker = reranker
        self.translations = translations or {}

    def search(self, query: str, k: int = 10, laws: Iterable[str] | None = None) -> list[Hit]:
        hits = self.base.search(query, k=max(k, self.reranker.depth), laws=laws)
        return self.reranker.rerank(query, hits, k, self.translations.get(query))


def question_scores(result: MethodResult, questions: Sequence[Question], metric: str) -> list[float]:
    """One value per question, for comparing two methods on the same questions."""
    if metric == "mrr@10":
        return [reciprocal_rank(result.per_question[q.id], q.gold) for q in questions]
    k = int(metric.split("@")[1])
    return [recall_at_k(result.per_question[q.id], q.gold, k) for q in questions]


def paired_bootstrap(
    baseline: Sequence[float], other: Sequence[float], rounds: int = 10000, seed: int = 0
) -> tuple[float, float, float]:
    """Mean of other - baseline with a 95% interval from resampling questions.
    An interval that spans 0 means the difference could be noise."""
    difference = np.asarray(other, dtype=np.float64) - np.asarray(baseline, dtype=np.float64)
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(difference), size=(rounds, len(difference)))
    means = difference[picks].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(difference.mean()), float(low), float(high)


def make_translations(
    questions: Sequence[Question], translator, path: Path = TRANSLATIONS_PATH, model: str = ""
) -> dict[str, str]:
    """Translate what is missing and save after each question, so a run that
    stops at a rate limit keeps what it has. Returns question text to translation."""
    saved = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"model": model, "by_id": {}}
    by_id = saved["by_id"]
    missing = [q for q in questions if q.id not in by_id]
    if missing and translator is None:
        raise RuntimeError(f"{len(missing)} questions have no saved translation and no LLM key is set")
    try:
        for q in missing:
            translation = translator.translate(q.question, q.lang)
            if translation:
                by_id[q.id] = translation
    finally:
        if missing:
            path.write_text(
                json.dumps(saved, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n"
            )
    return {q.question: by_id[q.id] for q in questions if q.id in by_id}


def _fmt(value: float) -> str:
    return f"{value:.3f}"


def render_markdown(
    results: Sequence[MethodResult],
    questions: Sequence[Question],
    articles: dict[str, Article],
    meta: dict[str, str],
    baseline: str | None = None,
    stages: Sequence[tuple[str, str]] = (),
) -> str:
    groups = subsets(questions, articles)
    lines = [
        "# Retrieval evaluation",
        "",
        f"Questions: {len(questions)} ({len(groups['en'])} English, {len(groups['ar'])} Arabic). "
        f"Articles indexed: {meta['articles']}. Generated by `qanun eval` on {meta['date']}.",
        "",
        "Recall@k is the share of gold articles found in the top k results, averaged over questions. "
        "MRR@10 is the mean reciprocal rank of the first gold article within the top 10.",
        "",
        "## All questions",
        "",
        "| Method | Recall@1 | Recall@3 | Recall@4 | Recall@5 | Recall@10 | MRR@10 |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in results:
        m = r.metrics(groups["all"])
        lines.append(
            f"| {r.method} | {_fmt(m['recall@1'])} | {_fmt(m['recall@3'])} | {_fmt(m['recall@4'])} "
            f"| {_fmt(m['recall@5'])} | {_fmt(m['recall@10'])} | {_fmt(m['mrr@10'])} |"
        )
    named = {r.method: r for r in results}

    def difference_row(label: str, old: MethodResult, new: MethodResult) -> str:
        cells = []
        for metric in ("recall@4", "recall@5", "mrr@10"):
            mean, low, high = paired_bootstrap(
                question_scores(old, questions, metric), question_scores(new, questions, metric)
            )
            cells.append(f"{mean:+.3f} ({low:+.3f} to {high:+.3f})")
        return f"| {label} | {' | '.join(cells)} |"

    interval_note = (
        "Mean difference with a 95% paired bootstrap interval over questions. "
        "An interval that spans 0 could be noise."
    )
    header = ["| Change | Recall@4 | Recall@5 | MRR@10 |", "| --- | --- | --- | --- |"]
    reference = named.get(baseline or "")
    if reference is not None and len(results) > 1:
        lines += ["", f"## Change from {reference.method}", "", interval_note, "", *header]
        lines += [difference_row(r.method, reference, r) for r in results if r is not reference]
    pairs = [(new, old) for new, old in stages if new in named and old in named]
    if pairs:
        lines += ["", "## What each stage adds", "", interval_note, "", *header]
        lines += [difference_row(f"{new}, against {old}", named[old], named[new]) for new, old in pairs]
    for name in ("en", "ar", "en, gold only in Arabic"):
        group = groups[name]
        lines += [
            "",
            f"## Subset: {name} ({len(group)} questions)",
            "",
            "| Method | Recall@1 | Recall@5 | MRR@10 |",
            "| --- | --- | --- | --- |",
        ]
        for r in results:
            m = r.metrics(group)
            if m:
                lines.append(f"| {r.method} | {_fmt(m['recall@1'])} | {_fmt(m['recall@5'])} | {_fmt(m['mrr@10'])} |")

    best = max(results, key=lambda r: r.metrics(groups["all"])["mrr@10"])
    misses = [q for q in questions if recall_at_k(best.per_question[q.id], q.gold, 5) < 1]
    lines += [
        "",
        f"## Misses for {best.method} (gold not fully in the top 5)",
        "",
    ]
    if not misses:
        lines.append("None.")
    else:
        lines += ["| Question | Gold | Top 3 returned |", "| --- | --- | --- |"]
        for q in misses:
            top = ", ".join(best.per_question[q.id][:3])
            lines.append(f"| {q.id}: {q.question} | {', '.join(q.gold)} | {top} |")
    lines.append("")
    return "\n".join(lines)


def run(
    models: Sequence[str],
    out_dir: Path = EVAL_DIR,
    translate: Callable[[Sequence[Question]], dict[str, str]] | None = None,
    rerankers: Sequence[str] = (),
) -> list[MethodResult]:
    from datetime import date

    from qanun.corpus.store import load_articles, load_laws
    from qanun.retrieval.bm25 import BM25Index
    from qanun.retrieval.dense import load_or_build
    from qanun.retrieval.hybrid import CrossLanguageRetriever, HybridRetriever

    articles_list = load_articles()
    articles = {a.id: a for a in articles_list}
    laws = {law.key: law for law in load_laws()}
    questions = load_questions()
    check_gold(questions, articles)

    bm25 = BM25Index.from_articles(articles_list, laws)
    results = [run_method("BM25", bm25, questions)]
    for model in models:
        dense = load_or_build(model, articles_list, laws)
        results.append(run_method(f"Dense ({model})", dense, questions))
        results.append(run_method(f"Hybrid RRF (BM25 + {model})", HybridRetriever([bm25, dense]), questions))
    baseline = f"Hybrid RRF (BM25 + {models[0]})"

    # The new stages are measured on the first model only.
    dense = load_or_build(models[0], articles_list, laws)
    cross = CrossLanguageRetriever(bm25, dense)
    translations = translate(questions) if translate else {}
    results.append(run_method("Cross-language, no translation", cross, questions))
    if translations:
        results.append(
            run_method("Cross-language + translation", TranslatedRetriever(cross, translations), questions)
        )
    stages = [("Cross-language + translation", "Cross-language, no translation")]
    for name in rerankers:
        from qanun.retrieval.rerank import CrossEncoderScorer, Reranker

        stages += [
            (f"Cross-language + {name}", "Cross-language, no translation"),
            (f"Cross-language + translation + {name}", "Cross-language + translation"),
        ]

        reranker = Reranker(CrossEncoderScorer(name), articles_list, laws)
        results.append(
            run_method(f"Cross-language + {name}", RerankedRetriever(cross, reranker), questions)
        )
        if translations:
            base = TranslatedRetriever(cross, translations)
            results.append(
                run_method(
                    f"Cross-language + translation + {name}",
                    RerankedRetriever(base, reranker, translations),
                    questions,
                )
            )

    meta = {
        "date": date.today().isoformat(),
        "articles": str(sum(1 for a in articles_list if a.in_force)),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    groups = subsets(questions, articles)
    payload = {
        "meta": meta,
        "methods": [
            {
                "method": r.method,
                "seconds": round(r.seconds, 3),
                "metrics": {name: r.metrics(group) for name, group in groups.items()},
                "rankings": r.per_question,
            }
            for r in results
        ],
    }
    (out_dir / "results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n"
    )
    (out_dir / "results.md").write_text(
        render_markdown(results, questions, articles, meta, baseline, stages),
        encoding="utf-8",
        newline="\n",
    )
    return results
