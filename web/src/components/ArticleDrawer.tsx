import { useEffect, useState } from "react";
import { api, splitId, type Article, type Lang } from "../api";
import { t } from "../i18n";

interface Props {
  articleId: string | null;
  uiLang: Lang;
  onClose: () => void;
}

export function ArticleDrawer({ articleId, uiLang, onClose }: Props) {
  const s = t(uiLang);
  const [article, setArticle] = useState<Article | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!articleId) return;
    const { law, label } = splitId(articleId);
    setArticle(null);
    setError(null);
    api.article(law, label).then(setArticle, (e: Error) => setError(e.message));
  }, [articleId]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  if (!articleId) return null;
  const ar = uiLang === "ar";

  return (
    <div className="drawer-backdrop" onClick={onClose}>
      <aside className="drawer" role="dialog" aria-modal="true" onClick={(e) => e.stopPropagation()}>
        <button className="close" onClick={onClose} aria-label={s.close}>
          &times;
        </button>
        {error && <p className="error">{error}</p>}
        {!article && !error && <p className="muted">...</p>}
        {article && (
          <>
            <p className="eyebrow">{ar ? article.law_name_ar : article.law_name_en}</p>
            <h2>{ar ? article.title_ar : article.title_en}</h2>
            <p className="muted small">
              {[ar ? article.chapter_ar : article.chapter_en ?? article.chapter_ar, ar ? article.section_ar : article.section_en ?? article.section_ar]
                .filter(Boolean)
                .join(" / ")}
            </p>
            <div className="tags">
              <span className={`badge ${article.status === "in_force" ? "badge-ok" : "badge-warn"}`}>
                {s.status[article.status] ?? article.status}
              </span>
              {article.effective_date && (
                <span className="badge">
                  {s.effective} {article.effective_date}
                </span>
              )}
            </div>

            <div className="texts">
              <section className="text-block" lang="ar" dir="rtl">
                <h4>{s.arabicOfficial}</h4>
                {article.text_ar.split("\n").map((line, i) => (
                  <p key={i}>{line}</p>
                ))}
              </section>
              <section className="text-block" lang="en" dir="ltr">
                <h4>{s.englishUnofficial}</h4>
                {article.text_en ? (
                  <>
                    {article.text_en.split("\n").map((line, i) => (
                      <p key={i}>{line}</p>
                    ))}
                    {article.text_en_source === "almeezan-legacy" && <p className="note small">{s.legacyNote}</p>}
                  </>
                ) : (
                  <p className="muted">{s.noEnglish}</p>
                )}
              </section>
            </div>

            {article.amendments.length > 0 && (
              <section>
                <h4>{s.amendments}</h4>
                <ul className="amendments">
                  {article.amendments.map((a) => (
                    <li key={`${a.instrument_ar}-${a.instrument_article}`}>
                      {a.change === "added" ? s.addedBy : s.amendedBy}{" "}
                      <strong>{ar ? a.instrument_ar : a.instrument_en ?? a.instrument_ar}</strong>
                      {a.instrument_article && ` (${s.article} ${a.instrument_article})`}
                      {a.effective_date && `, ${a.effective_date}`}
                    </li>
                  ))}
                </ul>
              </section>
            )}

            {article.notes.length > 0 && (
              <section>
                <h4>{s.notes}</h4>
                {article.notes.map((n) => (
                  <p key={n} className="note small" dir="ltr">
                    {n}
                  </p>
                ))}
              </section>
            )}

            {article.history.length > 0 && (
              <details className="history">
                <summary>
                  {s.previousVersions} ({article.history.length})
                </summary>
                {article.history.map((h, i) => (
                  <div key={i} className="text-block" lang="ar" dir="rtl">
                    <p className="muted small" dir="ltr">
                      {h.effective_from ?? "?"} to {h.effective_to ?? "?"}
                    </p>
                    {h.text_ar.split("\n").map((line, j) => (
                      <p key={j}>{line}</p>
                    ))}
                  </div>
                ))}
              </details>
            )}

            <p>
              <a href={article.source_url} target="_blank" rel="noreferrer">
                {s.openAlmeezan}
              </a>
            </p>
          </>
        )}
      </aside>
    </div>
  );
}
