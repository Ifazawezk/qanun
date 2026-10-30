"""Command line entry point: `qanun <command>`."""

from __future__ import annotations

import argparse
import logging
import time


def cmd_fetch(args: argparse.Namespace) -> None:
    from qanun.corpus import fetch

    fetch.main(["--refresh"] if args.refresh else [])


def cmd_parse(args: argparse.Namespace) -> None:
    from qanun.corpus import parse

    parse.main([])


def cmd_index(args: argparse.Namespace) -> None:
    from qanun.corpus.store import load_articles, load_laws
    from qanun.retrieval.dense import load_or_build

    articles = load_articles()
    laws = {law.key: law for law in load_laws()}
    for model in args.model:
        start = time.perf_counter()
        index = load_or_build(model, articles, laws, rebuild=args.rebuild)
        print(
            f"{model}: {len(index.passages)} passages, {index.embeddings.shape[1]} dimensions "
            f"({time.perf_counter() - start:.1f}s)"
        )


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
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = build_parser().parse_args(argv)
    if getattr(args, "model", "unset") is None:
        from qanun.retrieval.dense import DEFAULT_MODEL

        args.model = [DEFAULT_MODEL]
    args.func(args)


if __name__ == "__main__":
    main()
