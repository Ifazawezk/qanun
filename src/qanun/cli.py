"""Command line entry point: `qanun <command>`."""

from __future__ import annotations

import argparse
import logging
import sys
import time


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


def cmd_eval(args: argparse.Namespace) -> None:
    from qanun.evaluation import EVAL_DIR, load_questions, run
    from qanun.retrieval.dense import DEFAULT_MODEL

    results = run(args.model or [DEFAULT_MODEL])
    questions = load_questions()
    print(f"{'method':45} R@1    R@3    R@5    R@10   MRR@10")
    for r in results:
        m = r.metrics(questions)
        print(
            f"{r.method:45} {m['recall@1']:.3f}  {m['recall@3']:.3f}  {m['recall@5']:.3f}  "
            f"{m['recall@10']:.3f}  {m['mrr@10']:.3f}"
        )
    print(f"wrote {EVAL_DIR / 'results.md'}")


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

    p = sub.add_parser("serve", help="run the HTTP API (and the web UI if it is built)")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--reload", action="store_true")
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("eval", help="evaluate BM25, dense and hybrid retrieval")
    p.add_argument("--model", action="append", help="embedding model(s) to compare")
    p.set_defaults(func=cmd_eval)
    return parser


def main(argv: list[str] | None = None) -> None:
    # Windows consoles default to a legacy code page that cannot print Arabic.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    for noisy in ("httpx", "httpx2", "sentence_transformers", "huggingface_hub", "transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
