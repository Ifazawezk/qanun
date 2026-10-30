# Contributing to qanun

Thanks for helping. The most useful contributions are new laws, better evaluation questions and corrections to the data.

One rule comes before the others: **never add legal text or a translation that did not come from a citable source.** If a source is missing, leave the field empty.

## Set up

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[embeddings,dev]"
pytest
```

The tests run offline. They use a fake encoder and a fake LLM from `tests/conftest.py` and `tests/test_generate.py`, so they need no model download or API key.

For the web UI, run `qanun serve` in one terminal and `npm ci && npm run dev` in `web/` in another.

## Add a law

1. Find the law on [Al Meezan](https://almeezan.qa) and note its number, year and type. The type is in the page address, for example `/legislations/decree-law/2020/18/general`.
2. Add a `LawSource` to `LAWS` in `src/qanun/corpus/sources.py` with a short, stable `key` such as `domestic-workers-2017`:

   ```python
   LawSource(
       key="domestic-workers-2017",
       number=15,
       year=2017,
       type_code="law",
       name_en="Domestic Workers Law (Law No. 15 of 2017)",
       name_ar="قانون المستخدمين في المنازل (القانون رقم 15 لسنة 2017)",
   ),
   ```

3. Download and parse it. The fetcher only asks for what is missing, and it is slow on purpose, so please leave the delay alone:

   ```bash
   qanun fetch
   qanun parse
   ```

4. Check the result in `data/corpus/articles.jsonl`:
   - The number of articles matches the law. `totalAlmawadCounts` in `data/raw/<key>/legislation.ar.json` is a good reference.
   - Article labels look right, especially `bis` articles.
   - Repealed and superseded versions are not marked `in_force`.
   - Text in `text_en` really is English. The API falls back to Arabic when a translation is missing.
5. Rebuild the index with `qanun index`, then run `pytest`. `tests/test_corpus.py` checks ids, sources and the laws file.
6. Add evaluation questions for the new law, as described below, and report the change in numbers in your pull request.
7. Update the tables in "Data sources and terms" in `README.md`.

If a law needs a source other than Al Meezan, explain in the pull request why, and record the source in `src/qanun/corpus/sources.py` with a note on how far it can be trusted. See `LEGACY_ENGLISH` for an example.

## Add evaluation questions

Questions live in `eval/questions.jsonl`, one JSON object per line:

```json
{"id": "en-36", "lang": "en", "topic": "annual leave", "question": "Who decides when I take my annual leave?", "gold": ["labour-2004:80"]}
```

| Field | Meaning |
| --- | --- |
| `id` | `en-NN` or `ar-NN`, next free number |
| `lang` | `en` or `ar` |
| `topic` | Short label |
| `question` | Phrased the way a worker or HR officer would ask it |
| `gold` | Ids of the articles that answer the question, as `<law key>:<label>` |

To keep the numbers meaningful:

- **Write the question first and look at search results afterwards.** Do not pick wording because you know it retrieves well, and do not change a question because a method missed it.
- Read the whole article before adding it to `gold`. Use the article that actually answers the question, not one that only mentions the topic.
- Prefer everyday words over the law's wording. Those are the cases that matter.
- Only in-force articles can be gold. `qanun eval` stops with an error otherwise.

Then run the evaluation with both models and commit the new report:

```bash
qanun eval --model e5-small --model minilm
```

Update the tables in `README.md` from `eval/results.md`. Copy the numbers as they are, and if a change makes them worse, keep the result and say so. Please do not tune settings against this question set without saying so in the pull request.

## Report a wrong answer or data error

Open an issue with the question, what qanun returned, what you expected and the article on Al Meezan that shows it. If you found a difference between a translation and the Arabic text, add it to `REVIEW_NOTES` in `src/qanun/corpus/parse.py` so it appears on the article. Do not edit `data/corpus` by hand: it is rebuilt from the raw data, so your change would be lost.

## Pull requests

- Keep each pull request to one change, and add tests for new behaviour.
- Run `pytest` and, for UI changes, `npm run build` in `web/`.
- Use short imperative commit messages, such as `Add domestic workers law`.
- Do not commit model weights, indexes, `data/raw` or API keys.
- Keep user-facing text in both languages. UI strings are in `web/src/i18n.ts`.
- Prompts belong in `src/qanun/prompts`, not in Python strings.

By contributing you agree that your code is released under the MIT license. Law texts stay with their publishers.
