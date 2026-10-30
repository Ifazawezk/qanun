import json

import httpx

from qanun.corpus.fetch import Manifest, PoliteClient, fetch_law
from qanun.corpus.sources import LawSource

LAW = LawSource(
    key="demo",
    number=1,
    year=2020,
    type_code="law",
    name_en="Demo Law",
    name_ar="قانون تجريبي",
)


def _payload(data):
    return {"success": True, "message": "ok", "data": data}


def make_transport(calls):
    tree = {
        "id": "law-1",
        "sections": [
            {
                "id": "s1",
                "title": "Chapter One",
                "almawad": [
                    {"id": "a1", "title": "مادة 1 ", "isAmended": False, "isAddition": False},
                    {"id": "a2", "title": "مادة 2  معدلة", "isAmended": True, "isAddition": False},
                ],
                "sections": [],
            }
        ],
    }

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append((request.url.path, request.headers.get("accept-language")))
        path = request.url.path
        if path.endswith("/details-by-key"):
            assert request.url.params["number"] == "1"
            return httpx.Response(200, json=_payload({"id": "law-1"}))
        if path.endswith("/Legislation/law-1"):
            return httpx.Response(200, json=_payload(tree))
        if "/mada/" in path:
            return httpx.Response(200, json=_payload({"relatedAlmawad": []}))
        return httpx.Response(404, json={"success": False})

    return httpx.MockTransport(handler)


def test_fetch_law_writes_files_and_only_fetches_history_for_amended(tmp_path):
    calls = []
    client = PoliteClient(delay=0, transport=make_transport(calls))
    manifest = Manifest(tmp_path / "manifest.json")

    fetch_law(LAW, client, manifest, tmp_path, refresh=False)
    manifest.save()

    law_dir = tmp_path / "demo"
    assert (law_dir / "legislation.ar.json").exists()
    assert (law_dir / "legislation.en.json").exists()
    assert sorted(p.name for p in (law_dir / "history").iterdir()) == ["a2.ar.json", "a2.en.json"]

    langs = [lang for path, lang in calls if path.endswith("/Legislation/law-1")]
    assert sorted(langs) == ["ar", "en"]

    entries = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert entries["demo/lookup.json"]["params"]["typeCode"] == "law"
    assert len(entries["demo/legislation.ar.json"]["sha256"]) == 64


def test_fetch_law_uses_cache_on_second_run(tmp_path):
    calls = []
    client = PoliteClient(delay=0, transport=make_transport(calls))
    manifest = Manifest(tmp_path / "manifest.json")
    fetch_law(LAW, client, manifest, tmp_path, refresh=False)
    first = len(calls)
    fetch_law(LAW, client, manifest, tmp_path, refresh=False)
    assert len(calls) == first


def test_polite_client_retries_server_errors():
    attempts = []

    def handler(request):
        attempts.append(1)
        if len(attempts) < 2:
            return httpx.Response(503)
        return httpx.Response(200, json={"ok": True})

    client = PoliteClient(delay=0, retries=3, transport=httpx.MockTransport(handler))
    assert client.get("https://example.test/x").json() == {"ok": True}
    assert len(attempts) == 2
