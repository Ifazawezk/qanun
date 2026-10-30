"""Read and write the processed corpus (JSON Lines plus a Parquet copy)."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from qanun.config import CORPUS_DIR
from qanun.corpus.models import Article, Law

ARTICLES_FILE = "articles.jsonl"
LAWS_FILE = "laws.json"
PARQUET_FILE = "articles.parquet"


def save_corpus(laws: list[Law], articles: list[Article], out_dir: Path = CORPUS_DIR) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / ARTICLES_FILE).open("w", encoding="utf-8", newline="\n") as fh:
        for article in articles:
            fh.write(json.dumps(article.to_dict(), ensure_ascii=False) + "\n")
    (out_dir / LAWS_FILE).write_text(
        json.dumps([law.to_dict() for law in laws], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    write_parquet(articles, out_dir / PARQUET_FILE)


def write_parquet(articles: list[Article], path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    rows = []
    for article in articles:
        row = article.to_dict()
        for key in ("amendments", "history", "notes"):
            row[key] = json.dumps(row[key], ensure_ascii=False)
        rows.append(row)
    pq.write_table(pa.Table.from_pylist(rows), path)


def load_articles(path: Path | None = None) -> list[Article]:
    path = path or CORPUS_DIR / ARTICLES_FILE
    with path.open(encoding="utf-8") as fh:
        return [Article.from_dict(json.loads(line)) for line in fh if line.strip()]


def load_laws(path: Path | None = None) -> list[Law]:
    path = path or CORPUS_DIR / LAWS_FILE
    return [Law(**item) for item in json.loads(path.read_text(encoding="utf-8"))]


@lru_cache(maxsize=4)
def cached_articles(path: str | None = None) -> tuple[Article, ...]:
    return tuple(load_articles(Path(path) if path else None))
