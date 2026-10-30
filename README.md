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

## Retrieval evaluation

`eval/questions.jsonl` holds 60 questions written by hand (35 English, 25 Arabic) about gratuity, leave, working hours, probation, notice, minimum wage, changing employers, overtime and more, each with the article(s) that answer it. `qanun eval` scores every retriever on them; the full report, including the questions each method misses, is in [eval/results.md](eval/results.md).

| Method | Recall@1 | Recall@5 | Recall@10 | MRR@10 |
| --- | --- | --- | --- | --- |
| BM25 | 0.375 | 0.617 | 0.750 | 0.503 |
| Dense (e5-small) | 0.558 | 0.650 | 0.767 | 0.612 |
| Hybrid RRF (BM25 + e5-small) | 0.458 | 0.717 | 0.767 | 0.560 |
| Dense (minilm) | 0.308 | 0.658 | 0.692 | 0.433 |
| Hybrid RRF (BM25 + minilm) | 0.358 | 0.750 | 0.817 | 0.516 |

Dense retrieval with e5-small is the strongest at the top of the list, while the fused rankings find more gold articles within the first five or ten results. The weak spot is the 9 English questions whose answer only exists in Arabic (no English translation on Al Meezan): e5-small finds none of them in the top 5, the MiniLM paraphrase model finds 4 of 9. See the subset tables in the report.
