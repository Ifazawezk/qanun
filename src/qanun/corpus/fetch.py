"""Download the raw legislation data.

The Al Meezan web app loads legislation from a public JSON API, so we read the
same endpoints instead of scraping rendered pages. The downloader is slow on
purpose: one request at a time, a pause between requests, and every response
is cached under data/raw so a rerun only fetches what is missing.

Usage:
    python -m qanun.corpus.fetch            # fetch everything that is missing
    python -m qanun.corpus.fetch --refresh  # ignore the cache
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from qanun.config import RAW_DIR
from qanun.corpus.sources import API_BASE, LAWS, LEGACY_ENGLISH, LawSource

log = logging.getLogger("qanun.fetch")

USER_AGENT = "qanun-corpus/0.1 (+https://github.com/Ifazawezk/qanun)"
LANGUAGES = ("ar", "en")


class PoliteClient:
    """An httpx client that waits between requests and retries server errors."""

    def __init__(
        self,
        delay: float = 1.5,
        retries: int = 3,
        timeout: float = 60.0,
        transport: httpx.BaseTransport | None = None,
    ):
        self.delay = delay
        self.retries = retries
        self._last = 0.0
        self._client = httpx.Client(
            timeout=timeout,
            follow_redirects=True,
            headers={"User-Agent": USER_AGENT},
            transport=transport,
        )

    def get(self, url: str, **kwargs: Any) -> httpx.Response:
        for attempt in range(1, self.retries + 1):
            wait = self.delay - (time.monotonic() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()
            try:
                resp = self._client.get(url, **kwargs)
            except httpx.TransportError as exc:
                log.warning("attempt %d for %s failed: %s", attempt, url, exc)
            else:
                if resp.status_code < 500 and resp.status_code != 429:
                    resp.raise_for_status()
                    return resp
                log.warning("attempt %d for %s returned %d", attempt, url, resp.status_code)
            time.sleep(self.delay * 2**attempt)
        raise RuntimeError(f"giving up on {url}")

    def close(self) -> None:
        self._client.close()


class Manifest:
    """Records the URL, time and checksum of every downloaded file."""

    def __init__(self, path: Path):
        self.path = path
        self.entries: dict[str, dict[str, Any]] = {}
        if path.exists():
            self.entries = json.loads(path.read_text(encoding="utf-8"))

    def record(self, file: Path, url: str, params: dict[str, str] | None = None) -> None:
        rel = file.relative_to(self.path.parent).as_posix()
        self.entries[rel] = {
            "url": url,
            "params": params or {},
            "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "sha256": hashlib.sha256(file.read_bytes()).hexdigest(),
            "bytes": file.stat().st_size,
        }

    def save(self) -> None:
        self.path.write_text(
            json.dumps(self.entries, indent=2, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )


def _get_json(
    client: PoliteClient,
    manifest: Manifest,
    dest: Path,
    url: str,
    lang: str | None = None,
    params: dict[str, str] | None = None,
    refresh: bool = False,
) -> dict[str, Any]:
    if dest.exists() and not refresh:
        return json.loads(dest.read_text(encoding="utf-8"))
    headers = {"Accept": "application/json"}
    if lang:
        headers["Accept-Language"] = lang
    resp = client.get(url, params=params, headers=headers)
    payload = resp.json()
    if not payload.get("success", False):
        raise RuntimeError(f"{url} answered with: {payload.get('message')}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    manifest.record(dest, url, {**(params or {}), **({"lang": lang} if lang else {})})
    log.info("saved %s", dest)
    return payload


def iter_articles(sections: list[dict[str, Any]]):
    for section in sections:
        yield from section.get("almawad") or []
        yield from iter_articles(section.get("sections") or [])


def fetch_law(
    law: LawSource, client: PoliteClient, manifest: Manifest, raw_dir: Path, refresh: bool
) -> None:
    law_dir = raw_dir / law.key
    lookup = _get_json(
        client,
        manifest,
        law_dir / "lookup.json",
        f"{API_BASE}/Legislation/details-by-key",
        params=law.lookup_params,
        refresh=refresh,
    )
    legislation_id = lookup["data"]["id"]

    details = {}
    for lang in LANGUAGES:
        details[lang] = _get_json(
            client,
            manifest,
            law_dir / f"legislation.{lang}.json",
            f"{API_BASE}/Legislation/{legislation_id}",
            lang=lang,
            refresh=refresh,
        )

    # Only amended or added articles have a version history worth fetching.
    for article in iter_articles(details["ar"]["data"]["sections"]):
        if not (article.get("isAmended") or article.get("isAddition") or article.get("hasHistory")):
            continue
        for lang in LANGUAGES:
            _get_json(
                client,
                manifest,
                law_dir / "history" / f"{article['id']}.{lang}.json",
                f"{API_BASE}/Legislation/{legislation_id}/mada/{article['id']}",
                lang=lang,
                refresh=refresh,
            )

    if law.key in LEGACY_ENGLISH:
        dest = law_dir / "legacy-en.pdf"
        if refresh or not dest.exists():
            url = LEGACY_ENGLISH[law.key]
            resp = client.get(url)
            dest.write_bytes(resp.content)
            manifest.record(dest, url)
            log.info("saved %s", dest)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--law", action="append", help="only fetch these law keys")
    parser.add_argument("--refresh", action="store_true", help="ignore cached files")
    parser.add_argument("--delay", type=float, default=1.5, help="seconds between requests")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    args.raw_dir.mkdir(parents=True, exist_ok=True)
    manifest = Manifest(args.raw_dir / "manifest.json")
    client = PoliteClient(delay=args.delay)
    try:
        for law in LAWS:
            if args.law and law.key not in args.law:
                continue
            log.info("fetching %s", law.key)
            fetch_law(law, client, manifest, args.raw_dir, args.refresh)
            manifest.save()
    finally:
        manifest.save()
        client.close()


if __name__ == "__main__":
    main()
