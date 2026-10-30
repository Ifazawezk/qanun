import { useCallback, useEffect, useState, type FormEvent } from "react";
import { api, type Answer, type Health, type Lang } from "./api";
import { examples, t } from "./i18n";
import { AnswerView } from "./components/AnswerView";
import { ArticleDrawer } from "./components/ArticleDrawer";
import { SearchView } from "./components/SearchView";

type Tab = "ask" | "search";

// Shareable state lives in the URL: ?lang=ar&q=question, ?tab=search&s=query,
// &article=labour-2004:54 opens an article.
const params = new URLSearchParams(window.location.search);

function initialLang(): Lang {
  const fromUrl = params.get("lang");
  if (fromUrl === "ar" || fromUrl === "en") return fromUrl;
  try {
    const saved = localStorage.getItem("qanun.lang");
    if (saved === "ar" || saved === "en") return saved;
  } catch {
    // storage can be unavailable; fall through
  }
  return navigator.language.startsWith("ar") ? "ar" : "en";
}

function updateUrl(changes: Record<string, string | null>) {
  const next = new URLSearchParams(window.location.search);
  for (const [key, value] of Object.entries(changes)) {
    if (value) next.set(key, value);
    else next.delete(key);
  }
  const query = next.toString();
  window.history.replaceState(null, "", query ? `?${query}` : window.location.pathname);
}

export default function App() {
  const [lang, setLang] = useState<Lang>(initialLang);
  const [tab, setTab] = useState<Tab>(params.get("tab") === "search" ? "search" : "ask");
  const [question, setQuestion] = useState(params.get("q") ?? "");
  const [answer, setAnswer] = useState<Answer | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [health, setHealth] = useState<Health | null>(null);
  const [openArticle, setOpenArticle] = useState<string | null>(params.get("article"));
  const s = t(lang);

  useEffect(() => {
    document.documentElement.lang = lang;
    document.documentElement.dir = lang === "ar" ? "rtl" : "ltr";
    try {
      localStorage.setItem("qanun.lang", lang);
    } catch {
      // ignore
    }
  }, [lang]);

  useEffect(() => {
    api.health().then(setHealth, () => setHealth(null));
  }, []);

  const ask = useCallback(async (text: string) => {
    const q = text.trim();
    if (!q) return;
    setQuestion(q);
    setBusy(true);
    setError(null);
    updateUrl({ q, tab: null, s: null });
    try {
      setAnswer(await api.ask(q));
    } catch (err) {
      setError((err as Error).message);
      setAnswer(null);
    } finally {
      setBusy(false);
    }
  }, []);

  useEffect(() => {
    const q = params.get("q");
    if (q && tab === "ask") ask(q);
    // run once on load
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const open = useCallback((id: string | null) => {
    setOpenArticle(id);
    updateUrl({ article: id });
  }, []);

  function onSubmit(e: FormEvent) {
    e.preventDefault();
    ask(question);
  }

  return (
    <div className="page">
      <header className="topbar">
        <div className="brand">
          <img className="logo" src="/favicon.svg" alt="" width="48" height="48" />
          <div>
            <h1>
              qanun <span className="brand-ar">قانون</span>
            </h1>
            <p className="tagline">{s.tagline}</p>
          </div>
        </div>
        <button className="lang-toggle" onClick={() => setLang(lang === "en" ? "ar" : "en")}>
          {s.switchTo}
        </button>
      </header>

      <p className="disclaimer" role="note">
        {s.disclaimer}
      </p>

      <nav className="tabs" role="tablist">
        {(["ask", "search"] as Tab[]).map((name) => (
          <button
            key={name}
            role="tab"
            aria-selected={tab === name}
            className={tab === name ? "active" : ""}
            onClick={() => {
              setTab(name);
              updateUrl({ tab: name === "search" ? "search" : null });
            }}
          >
            {name === "ask" ? s.tabAsk : s.tabSearch}
          </button>
        ))}
        {health && (
          <span className="status muted small">
            {s.results(health.articles)} · {health.embedding_model ?? "BM25"} · {health.llm ?? s.extractiveBadge}
          </span>
        )}
      </nav>

      {tab === "ask" ? (
        <main>
          <form className="ask-form" onSubmit={onSubmit}>
            <textarea
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              placeholder={s.placeholder}
              rows={2}
              dir="auto"
              aria-label={s.ask}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  ask(question);
                }
              }}
            />
            <button className="primary" disabled={busy}>
              {busy ? s.asking : s.ask}
            </button>
          </form>
          <div className="examples">
            <span className="muted small">{s.examples}:</span>
            {examples[lang].map((ex) => (
              <button key={ex} className="chip" onClick={() => ask(ex)} dir="auto">
                {ex}
              </button>
            ))}
          </div>
          <p className="muted small">{s.answerLanguage}</p>
          {error && (
            <p className="error">
              {s.error}: {error}
            </p>
          )}
          {answer && <AnswerView answer={answer} uiLang={lang} onOpen={open} />}
        </main>
      ) : (
        <main>
          <SearchView
            key={(health?.search_methods ?? []).join()}
            uiLang={lang}
            methods={health?.search_methods ?? ["bm25"]}
            onOpen={open}
            initialQuery={params.get("s") ?? ""}
          />
        </main>
      )}

      <footer className="footer muted small">{s.footer}</footer>
      <ArticleDrawer articleId={openArticle} uiLang={lang} onClose={() => open(null)} />
    </div>
  );
}
