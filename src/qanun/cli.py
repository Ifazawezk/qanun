"""Command line entry point: `qanun <command>`."""

from __future__ import annotations

import argparse
import logging
import sys
import time

from qanun.env import load_env_file


def cmd_fetch(args: argparse.Namespace) -> None:
    from qanun.corpus import fetch

    fetch.main(["--refresh"] if args.refresh else [])


def cmd_parse(args: argparse.Namespace) -> None:
    from qanun.corpus import parse

    parse.main([])


def cmd_index(args: argparse.Namespace) -> None:
    from qanun.corpus.store import load_articles, load_laws
    from qanun.retrieval.dense import DEFAULT_MODEL, load_or_build

    from qanun.app_factory import embedding_model

    articles = load_articles()
    laws = {law.key: law for law in load_laws()}
    for model in args.model or [embedding_model() or DEFAULT_MODEL]:
        start = time.perf_counter()
        index = load_or_build(model, articles, laws, rebuild=args.rebuild)
        print(
            f"{model}: {len(index.passages)} passages, {index.embeddings.shape[1]} dimensions "
            f"({time.perf_counter() - start:.1f}s)"
        )


def eval_translator(budget_requests: int):
    """Translations for the evaluation, spending at most the given number of requests."""
    from qanun.answer.cache import Budget, CachedClient, ResponseCache
    from qanun.answer.llm import client_from_env
    from qanun.answer.translate import LLMTranslator
    from qanun.evaluation import CACHE_PATH, TRANSLATIONS_PATH, make_translations

    client = client_from_env(fast=True, max_wait=70)
    cached = None
    if client:
        cached = CachedClient(client, ResponseCache(CACHE_PATH), Budget(requests=budget_requests))

    def translate(questions):
        translator = LLMTranslator(cached) if cached else None
        return make_translations(questions, translator, TRANSLATIONS_PATH, client.model if client else "")

    return translate


def cmd_eval(args: argparse.Namespace) -> None:
    from qanun.evaluation import EVAL_DIR, load_questions, run
    from qanun.retrieval.dense import DEFAULT_MODEL

    translate = eval_translator(args.budget) if args.translate else None
    results = run(args.model or [DEFAULT_MODEL], translate=translate, rerankers=args.reranker or ())
    questions = load_questions()
    print(f"{'method':45} R@1    R@3    R@4    R@5    R@10   MRR@10")
    for r in results:
        m = r.metrics(questions)
        print(
            f"{r.method:45} {m['recall@1']:.3f}  {m['recall@3']:.3f}  {m['recall@4']:.3f}  "
            f"{m['recall@5']:.3f}  {m['recall@10']:.3f}  {m['mrr@10']:.3f}"
        )
    print(f"wrote {EVAL_DIR / 'results.md'}")


def cmd_eval_answers(args: argparse.Namespace) -> None:
    from qanun import answer_eval
    from qanun.answer.assistant import Assistant
    from qanun.answer.cache import Budget, CachedClient, ResponseCache
    from qanun.answer.generate import Generator
    from qanun.answer.llm import client_from_env
    from qanun.answer.translate import LLMTranslator
    from qanun.app_factory import embedding_model, reranker_model
    from qanun.evaluation import CACHE_PATH
    from qanun.search import SearchEngine

    # Long enough to sit out the per-minute window, never long enough to outlast a daily limit.
    client = client_from_env(model=args.model, max_wait=70, max_output=args.max_tokens)
    if client is None:
        sys.exit("No LLM key is set, so the answer evaluation was not run and no numbers were written.")
    cache = ResponseCache(CACHE_PATH)
    budget = Budget(requests=args.budget, tokens=args.tokens, daily_tokens=args.daily_tokens)
    answers = CachedClient(client, cache, budget)
    translator = LLMTranslator(CachedClient(client_from_env(fast=True, max_wait=70), cache, budget))
    judge = None
    judge_model = None
    if not args.no_judge:
        judge_client = client_from_env(fast=True, model=args.judge_model, max_wait=70)
        judge, judge_model = CachedClient(judge_client, cache, budget), judge_client.model

    embedding = embedding_model()
    engine = SearchEngine.load(embedding, reranker_model(embedding, True))
    generator = Generator(answers, engine.laws, args.max_article_chars)
    assistant = Assistant(engine, generator, translator=translator)
    items = answer_eval.choose(answer_eval.load_items(), args.sample, args.lang)
    name = args.name or answer_eval.slug(f"{client.model}-{args.lang or 'all'}")
    meta = {
        "name": name, "provider": client.provider, "model": client.model, "judge_model": judge_model,
        "language": args.lang or "all", "max_article_chars": args.max_article_chars,
    }
    payload = answer_eval.run(items, assistant, answers, judge, answer_eval.RESULTS_DIR / f"{name}.json", meta)
    answer_eval.write_report()
    m = payload["meta"]
    print(f"{name}: finished {m['finished']} of {m['planned']} questions")
    print(f"this run: {m['requests_this_run']} requests, {m['tokens_this_run']} tokens, {m['cache_hits_this_run']} cached")
    if m["stopped_by"]:
        print(f"stopped: {m['stopped_by']}. Run the same command later to resume.")
    print(f"wrote {answer_eval.REPORT_PATH}")


def cmd_llm(args: argparse.Namespace) -> None:
    """Show the LLM settings. It reports whether a key is set, and never the key."""
    import os

    from qanun.answer.llm import client_from_env, model_from_env

    client = client_from_env()
    print(f"key set: {'yes' if client else 'no'}")
    print(f"provider: {client.provider if client else 'none'}")
    print(f"answer model: {model_from_env() or 'none'}")
    print(f"fast model: {model_from_env(fast=True) or 'none'}")
    print(f"translation: {os.environ.get('QANUN_TRANSLATE', 'on') if client else 'off'}")


def cmd_ask(args: argparse.Namespace) -> None:
    from qanun.answer.highlight import render_marked
    from qanun.app_factory import build_assistant

    assistant = build_assistant(model=args.model[0] if args.model else None)
    answer = assistant.ask(" ".join(args.question), k=args.k, mode=args.mode)
    print(answer.text)
    print()
    for c in answer.citations:
        print(f"[{c.rank}] {c.title} - {c.law_name}")
        print(f"    {c.url}")
        if c.passage:
            marked = render_marked(c.passage)
            print("    " + marked.replace(chr(10), chr(10) + "    "))
        print()


def cmd_serve(args: argparse.Namespace) -> None:
    import uvicorn

    uvicorn.run("qanun.api.app:app", host=args.host, port=args.port, reload=args.reload)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="qanun", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("fetch", help="download raw legislation from Al Meezan")
    p.add_argument("--refresh", action="store_true")
    p.set_defaults(func=cmd_fetch)

    p = sub.add_parser("parse", help="build data/corpus from the raw downloads")
    p.set_defaults(func=cmd_parse)

    p = sub.add_parser("index", help="embed the corpus and save the vector index")
    p.add_argument("--model", action="append", help="preset (e5-small, minilm) or model id")
    p.add_argument("--rebuild", action="store_true")
    p.set_defaults(func=cmd_index)

    p = sub.add_parser("ask", help="ask a question from the command line")
    p.add_argument("question", nargs="+")
    p.add_argument("-k", type=int, default=4, help="number of articles to return")
    p.add_argument("--mode", default="auto", choices=["auto", "extractive", "generative"])
    p.add_argument("--model", action="append", help="embedding model")
    p.set_defaults(func=cmd_ask)

    p = sub.add_parser("llm", help="show the LLM settings without revealing the key")
    p.set_defaults(func=cmd_llm)

    p = sub.add_parser("eval-answers", help="evaluate generated answers against a real model")
    p.add_argument("--sample", type=int, help="questions to run; a larger sample keeps the smaller one")
    p.add_argument("--lang", choices=["en", "ar"], help="only questions in this language")
    p.add_argument("--model", help="answer model to compare")
    p.add_argument("--judge-model", help="judge model; defaults to the fast model")
    p.add_argument("--no-judge", action="store_true", help="skip the LLM judge")
    p.add_argument("--name", help="name of this run in the report")
    p.add_argument("--budget", type=int, default=40, help="most requests this run may make")
    p.add_argument("--tokens", type=int, default=60000, help="most tokens this run may spend")
    p.add_argument("--daily-tokens", type=int, default=180000, help="most tokens per model per day")
    p.add_argument("--max-article-chars", type=int, help="cut articles, for small context windows")
    p.add_argument("--max-tokens", type=int, help="reply limit for the answer model only")
    p.set_defaults(func=cmd_eval_answers)

    p = sub.add_parser("serve", help="run the HTTP API (and the web UI if it is built)")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--reload", action="store_true")
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("eval", help="evaluate BM25, dense and hybrid retrieval")
    p.add_argument("--model", action="append", help="embedding model(s) to compare")
    p.add_argument("--translate", action="store_true", help="also measure query translation")
    p.add_argument("--reranker", action="append", help="reranker(s) to measure: mmarco or a model id")
    p.add_argument("--budget", type=int, default=70, help="most LLM requests to spend on translation")
    p.set_defaults(func=cmd_eval)
    return parser


def main(argv: list[str] | None = None) -> None:
    # Windows consoles default to a legacy code page that cannot print Arabic.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    load_env_file()
    for noisy in ("httpx", "httpx2", "sentence_transformers", "huggingface_hub", "transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
