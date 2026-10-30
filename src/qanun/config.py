"""Paths and settings shared across the package."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(os.environ.get("QANUN_ROOT", Path(__file__).resolve().parents[2]))
DATA_DIR = Path(os.environ.get("QANUN_DATA_DIR", ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"
CORPUS_DIR = DATA_DIR / "corpus"
INDEX_DIR = Path(os.environ.get("QANUN_INDEX_DIR", DATA_DIR / "index"))
ARTICLES_PATH = CORPUS_DIR / "articles.jsonl"
