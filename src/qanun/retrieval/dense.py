"""Dense retrieval with multilingual sentence embeddings.

Arabic and English passages share one vector space, so an English question can
land on an article that only exists in Arabic and the other way round. The
index is a plain normalized numpy matrix: the corpus has a few hundred
articles, so an exact dot product is fast and needs no approximate search.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Protocol, Sequence

import numpy as np

from qanun.corpus.models import Article, Law
from qanun.retrieval.base import Hit, law_mask, top_hits
from qanun.retrieval.documents import Passage, article_passages

log = logging.getLogger("qanun.dense")


@dataclass(frozen=True)
class ModelSpec:
    model_name: str
    query_prefix: str = ""
    passage_prefix: str = ""
    max_chars: int = 900  # passage size; keep it under the model's token limit


MODEL_PRESETS = {
    "e5-small": ModelSpec("intfloat/multilingual-e5-small", "query: ", "passage: ", 900),
    "minilm": ModelSpec(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", max_chars=400
    ),
}
DEFAULT_MODEL = "e5-small"


def resolve_model(name: str) -> ModelSpec:
    return MODEL_PRESETS.get(name) or ModelSpec(name)


class Encoder(Protocol):
    name: str
    max_chars: int

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray: ...

    def encode_passages(self, texts: Sequence[str]) -> np.ndarray: ...


class SentenceTransformerEncoder:
    """Wraps a sentence-transformers model and returns L2-normalized vectors."""

    def __init__(self, spec: ModelSpec, device: str | None = None, batch_size: int = 32):
        from sentence_transformers import SentenceTransformer

        self.spec = spec
        self.name = spec.model_name
        self.max_chars = spec.max_chars
        self.batch_size = batch_size
        self.model = SentenceTransformer(spec.model_name, device=device)

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        vectors = self.model.encode(
            list(texts),
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vectors.astype(np.float32)

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode([self.spec.query_prefix + t for t in texts])

    def encode_passages(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode([self.spec.passage_prefix + t for t in texts])


def corpus_fingerprint(passages: Sequence[Passage]) -> str:
    digest = hashlib.sha256()
    for p in passages:
        digest.update(f"{p.article_id}\t{p.lang}\t{p.text}\n".encode("utf-8"))
    return digest.hexdigest()[:16]


class DenseIndex:
    name = "dense"

    def __init__(
        self,
        encoder: Encoder,
        passages: list[Passage],
        embeddings: np.ndarray,
        article_laws: dict[str, str],
    ):
        self.encoder = encoder
        self.passages = passages
        self.embeddings = embeddings
        self.doc_ids = list(dict.fromkeys(p.article_id for p in passages))
        self.doc_laws = [article_laws.get(d, "") for d in self.doc_ids]
        position = {doc: i for i, doc in enumerate(self.doc_ids)}
        self._passage_doc = np.array([position[p.article_id] for p in passages], dtype=np.int64)
        self._passage_lang = np.array([p.lang for p in passages])

    @staticmethod
    def make_passages(
        articles: Iterable[Article], laws: dict[str, Law] | None, max_chars: int
    ) -> list[Passage]:
        laws = laws or {}
        passages: list[Passage] = []
        for article in articles:
            if article.in_force:
                passages.extend(article_passages(article, laws.get(article.law), max_chars))
        return passages

    @classmethod
    def build(
        cls,
        articles: Iterable[Article],
        encoder: Encoder,
        laws: dict[str, Law] | None = None,
    ) -> "DenseIndex":
        articles = list(articles)
        passages = cls.make_passages(articles, laws, encoder.max_chars)
        log.info("embedding %d passages with %s", len(passages), encoder.name)
        embeddings = encoder.encode_passages([p.text for p in passages])
        return cls(encoder, passages, embeddings, {a.id: a.law for a in articles})

    # ------------------------------------------------------------ persistence

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "embeddings.npy", self.embeddings)
        (directory / "passages.jsonl").write_text(
            "".join(
                json.dumps({"article_id": p.article_id, "lang": p.lang, "text": p.text}, ensure_ascii=False)
                + "\n"
                for p in self.passages
            ),
            encoding="utf-8",
        )
        meta = {
            "model": self.encoder.name,
            "dimensions": int(self.embeddings.shape[1]),
            "passages": len(self.passages),
            "fingerprint": corpus_fingerprint(self.passages),
        }
        (directory / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    @classmethod
    def load(
        cls,
        directory: Path,
        encoder: Encoder,
        articles: Iterable[Article],
        laws: dict[str, Law] | None = None,
    ) -> "DenseIndex":
        """Load a saved index, refusing one built from another model or corpus."""
        articles = list(articles)
        meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
        if meta["model"] != encoder.name:
            raise ValueError(f"index was built with {meta['model']}, not {encoder.name}")
        passages = cls.make_passages(articles, laws, encoder.max_chars)
        if meta["fingerprint"] != corpus_fingerprint(passages):
            raise ValueError("index is out of date with the corpus; rebuild it")
        embeddings = np.load(directory / "embeddings.npy")
        return cls(encoder, passages, embeddings, {a.id: a.law for a in articles})

    # ------------------------------------------------------------ search

    def passage_scores(self, query: str) -> np.ndarray:
        vector = self.encoder.encode_queries([query])[0]
        return self.embeddings @ vector

    def scores(self, query: str, lang: str | None = None) -> np.ndarray:
        """Article score = best cosine similarity among its passages, optionally
        only those in one language. An article with none scores -inf."""
        per_passage = self.passage_scores(query)
        if lang is not None:
            per_passage = np.where(self._passage_lang == lang, per_passage, -np.inf)
        scores = np.full(len(self.doc_ids), -np.inf)
        np.maximum.at(scores, self._passage_doc, per_passage)
        return scores

    def search(
        self, query: str, k: int = 10, laws: Iterable[str] | None = None, lang: str | None = None
    ) -> list[Hit]:
        # Fusion only sees ranks, so in a per-language list a passage with no similarity
        # at all must be left out, or it would earn the credit of a real match.
        min_score = 0.0 if lang else None
        return top_hits(self.doc_ids, self.scores(query, lang), k, law_mask(self.doc_laws, laws), min_score)


def index_dir_for(model: str, root: Path | None = None) -> Path:
    from qanun.config import INDEX_DIR

    return (root or INDEX_DIR) / model.replace("/", "__")


def load_or_build(
    model: str,
    articles: Sequence[Article],
    laws: dict[str, Law] | None = None,
    root: Path | None = None,
    rebuild: bool = False,
    encoder: Encoder | None = None,
) -> DenseIndex:
    """Load the saved index for a model, building and saving it when needed."""
    encoder = encoder or SentenceTransformerEncoder(resolve_model(model))
    directory = index_dir_for(model, root)
    if not rebuild and (directory / "meta.json").exists():
        try:
            return DenseIndex.load(directory, encoder, articles, laws)
        except ValueError as exc:
            log.warning("rebuilding %s: %s", directory, exc)
    index = DenseIndex.build(articles, encoder, laws)
    index.save(directory)
    return index
