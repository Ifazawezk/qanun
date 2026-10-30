import { useEffect, useState, type FormEvent } from "react";
import { api, type Lang, type SearchResult } from "../api";
import { t } from "../i18n";
import { Highlighted } from "./Highlighted";

interface Props {
  uiLang: Lang;
  methods: string[];
  onOpen: (articleId: string) => void;
  initialQuery?: string;
}

export function SearchView({ uiLang, methods, onOpen, initialQuery = "" }: Props) {
  const s = t(uiLang);
  const [query, setQuery] = useState(initialQuery);
  const [method, setMethod] = useState(methods.includes("hybrid") ? "hybrid" : methods[0] ?? "bm25");
  const [result, setResult] = useState<SearchResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function run(e?: FormEvent) {
    e?.preventDefault();
    if (!query.trim()) return;
    setBusy(true);
    setError(null);
    try {
      setResult(await api.search(query.trim(), method));
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusy(false);
    }
  }

  useEffect(() => {
    if (initialQuery) run();
    // only on first render
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <section>
      <form className="search-form" onSubmit={run}>
        <input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder={s.placeholder}
          dir="auto"
          aria-label={s.search}
        />
        <label className="method">
          <span>{s.method}</span>
          <select value={method} onChange={(e) => setMethod(e.target.value)}>
            {methods.map((m) => (
              <option key={m} value={m}>
                {s.methods[m] ?? m}
              </option>
            ))}
          </select>
        </label>
        <button className="primary" disabled={busy}>
          {busy ? s.asking : s.search}
        </button>
      </form>
      {error && <p className="error">{error}</p>}
      {result && (
        <>
          <p className="muted small">{s.results(result.hits.length)}</p>
          <ol className="hits">
            {result.hits.map((hit) => (
              <li key={hit.article_id} className="citation">
                <header>
                  <span className="rank">{hit.rank}</span>
                  <button className="link-button citation-title" onClick={() => onOpen(hit.article_id)}>
                    {uiLang === "ar" ? hit.title_ar : hit.title_en}
                  </button>
                  <span className="law-name">{hit.law}</span>
                  <span className="score">{hit.score.toFixed(3)}</span>
                </header>
                {hit.passage && <Highlighted passage={hit.passage} />}
              </li>
            ))}
          </ol>
        </>
      )}
    </section>
  );
}
