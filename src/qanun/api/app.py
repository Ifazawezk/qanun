"""FastAPI service.

    uvicorn qanun.api.app:app            # or: qanun serve

Endpoints: GET /health, GET /laws, POST /ask, GET /search, GET /articles,
GET /articles/{label}. When the web UI has been built (web/dist), it is served
from the same origin at /.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Callable

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from qanun.answer.assistant import Assistant
from qanun.answer.highlight import best_passage
from qanun.api.schemas import (
    AnswerOut,
    ArticleOut,
    ArticleSummary,
    AskRequest,
    HealthOut,
    LawOut,
    SearchHit,
    SearchOut,
)
from qanun.config import ROOT
from qanun.corpus.sources import DEFAULT_LAW
from qanun.text.normalize import detect_language

DEFAULT_STATIC = ROOT / "web" / "dist"


class DropQueryStrings(logging.Filter):
    """Keep questions out of the access log. /search carries one in its URL, and a
    question can hold personal details such as a name, salary or employer."""

    def filter(self, record: logging.LogRecord) -> bool:
        args = record.args
        if isinstance(args, tuple) and len(args) >= 3 and isinstance(args[2], str):
            record.args = (*args[:2], args[2].split("?", 1)[0], *args[3:])
        return True


logging.getLogger("uvicorn.access").addFilter(DropQueryStrings())


def _assistant(request: Request) -> Assistant:
    return request.app.state.assistant


def create_app(
    assistant: Assistant | None = None,
    factory: Callable[[], Assistant] | None = None,
    static_dir: Path | None = None,
) -> FastAPI:
    """Pass a ready assistant (tests) or a factory that is called on startup."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if getattr(app.state, "assistant", None) is None:
            from qanun.app_factory import build_assistant

            app.state.assistant = (factory or build_assistant)()
        yield

    app = FastAPI(
        title="qanun",
        version="0.1.0",
        description="Bilingual search and question answering over Qatar's Labour Law. Not legal advice.",
        lifespan=lifespan,
    )
    app.state.assistant = assistant

    origins = [o for o in os.environ.get("QANUN_CORS_ORIGINS", "").split(",") if o]
    if origins:
        app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["GET", "POST"], allow_headers=["*"])

    @app.get("/health", response_model=HealthOut)
    def health(request: Request) -> HealthOut:
        a = _assistant(request)
        dense = a.engine.dense
        generator = a.generator
        return HealthOut(
            status="ok",
            articles=sum(1 for x in a.engine.articles.values() if x.in_force),
            search_methods=a.engine.methods,
            default_method=a.engine.default_method,
            embedding_model=dense.encoder.name if dense else None,
            reranker=a.engine.reranker.scorer.name if a.engine.reranker else None,
            generation=a.can_generate,
            translation=a.translator is not None,
            llm=f"{generator.client.provider}:{generator.client.model}" if generator else None,
        )

    @app.get("/laws", response_model=list[LawOut])
    def laws(request: Request) -> list[LawOut]:
        return [LawOut(**law.to_dict()) for law in _assistant(request).engine.laws.values()]

    @app.post("/ask", response_model=AnswerOut)
    def ask(body: AskRequest, request: Request) -> AnswerOut:
        a = _assistant(request)
        if body.mode == "generative" and not a.can_generate:
            raise HTTPException(400, "generative mode needs an LLM API key on the server")
        _check_laws(a, body.laws)
        answer = a.ask(body.question, k=body.k, laws=body.laws, mode=body.mode)
        return AnswerOut(**answer.to_dict())

    @app.get("/search", response_model=SearchOut)
    def search(
        request: Request,
        q: str = Query(min_length=1, max_length=1000),
        k: int = Query(10, ge=1, le=50),
        method: str | None = Query(None, description="bm25, dense, hybrid or cross"),
        law: list[str] | None = Query(None),
    ) -> SearchOut:
        a = _assistant(request)
        _check_laws(a, law)
        try:
            hits = a.engine.search(q, k=k, method=method, laws=law)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        lang = detect_language(q)
        out = []
        for hit in hits:
            article = a.engine.article(hit.article_id)
            text_lang = "en" if lang == "en" and article.text_en else "ar"
            passage = best_passage(q, article.text(text_lang), text_lang, idf=a.engine.bm25.idf)
            out.append(
                SearchHit(
                    rank=hit.rank,
                    score=hit.score,
                    article_id=article.id,
                    law=article.law,
                    label=article.label,
                    title_en=article.title_en,
                    title_ar=article.title_ar,
                    passage=asdict(passage),
                )
            )
        return SearchOut(query=q, method=method or a.engine.default_method, hits=out)

    @app.get("/articles", response_model=list[ArticleSummary])
    def list_articles(request: Request, law: str = DEFAULT_LAW) -> list[ArticleSummary]:
        a = _assistant(request)
        _check_laws(a, [law])
        items = sorted(
            (x for x in a.engine.articles.values() if x.law == law),
            key=lambda x: x.order,
        )
        return [
            ArticleSummary(
                id=x.id, law=x.law, label=x.label, title_en=x.title_en, title_ar=x.title_ar,
                section_en=x.section_en, section_ar=x.section_ar, status=x.status,
                has_english=bool(x.text_en),
            )
            for x in items
        ]

    @app.get("/articles/{label}", response_model=ArticleOut)
    def get_article(label: str, request: Request, law: str = DEFAULT_LAW) -> ArticleOut:
        """An article by its label within a law: 54, 52-bis, 115-bis-1, issuance-2."""
        a = _assistant(request)
        _check_laws(a, [law])
        article = a.engine.find(law, label.strip().lower().replace(" ", "-"))
        if article is None:
            raise HTTPException(404, f"article {label} not found in {law}")
        law_obj = a.engine.laws[law]
        data = article.to_dict()
        return ArticleOut(**data, law_name_en=law_obj.name_en, law_name_ar=law_obj.name_ar)

    static = static_dir if static_dir is not None else Path(os.environ.get("QANUN_STATIC_DIR", DEFAULT_STATIC))
    if static and (static / "index.html").exists():
        app.mount("/", StaticFiles(directory=static, html=True), name="ui")

    return app


def _check_laws(assistant: Assistant, laws: list[str] | None) -> None:
    unknown = [law for law in laws or [] if law not in assistant.engine.laws]
    if unknown:
        raise HTTPException(400, f"unknown law key(s): {', '.join(unknown)}")


app = create_app()
