# qanun

Question answering over Qatar's Labour Law in Arabic and English, with answers that cite the exact articles.

Work in progress. The first step is a polite downloader for the official legislation text published on Al Meezan (almeezan.qa):

```bash
python -m venv .venv
.venv/Scripts/activate        # or: source .venv/bin/activate
pip install -e ".[dev]"
python -m qanun.corpus.fetch
```
