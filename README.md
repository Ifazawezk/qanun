# qanun

Question answering over Qatar's Labour Law in Arabic and English, with answers that cite the exact articles.

Work in progress.

## Corpus

The text comes from Al Meezan (almeezan.qa), the legal portal of Qatar's Ministry of Justice. `data/corpus/articles.jsonl` holds one record per article with the Arabic text, the English translation where one exists, chapter, amendment history and a link back to the article on Al Meezan.

| Law | Articles in force | With English |
| --- | --- | --- |
| Labour Law (Law No. 14 of 2004, as amended) | 163 | 146 |
| Minimum Wage Law (Law No. 17 of 2020) | 8 | 0 |
| Ministerial Decision No. 25 of 2020 (minimum wage amounts) | 5 | 0 |
| Expatriates Entry, Exit and Residency Law (Law No. 21 of 2015) | 43 | 43 |

To rebuild it from scratch:

```bash
python -m venv .venv
.venv/Scripts/activate        # or: source .venv/bin/activate
pip install -e ".[dev]"
python -m qanun.corpus.fetch  # downloads into data/raw, slowly and with caching
python -m qanun.corpus.parse  # writes data/corpus/articles.jsonl, laws.json and a Parquet copy
```
