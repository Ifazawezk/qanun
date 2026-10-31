import type { Answer, Citation, Lang } from "../api";
import { t } from "../i18n";
import { Highlighted } from "./Highlighted";

const CITATION = /\[([a-z0-9-]+:[a-z0-9-]+)\]/g;

interface Props {
  answer: Answer;
  uiLang: Lang;
  onOpen: (articleId: string) => void;
}

/**
 * Answer text with [law:label] markers turned into buttons that open the article.
 * The server has already validated them. A marker that is not among the returned
 * articles is dropped here as well, so it can never become a link.
 */
function AnswerText({ answer, uiLang, onOpen }: { answer: Answer; uiLang: Lang; onOpen: (id: string) => void }) {
  const s = t(uiLang);
  const byId = new Map(answer.citations.map((c) => [c.article_id, c]));
  const lines = answer.text.split(/\n+/).filter((line) => line.trim());
  return (
    <div className="answer-text" dir={answer.language === "ar" ? "rtl" : "ltr"} lang={answer.language}>
      {lines.map((line, li) => {
        const pieces: (string | JSX.Element)[] = [];
        let last = 0;
        for (const match of line.matchAll(CITATION)) {
          const index = match.index ?? 0;
          pieces.push(line.slice(last, index));
          const citation = byId.get(match[1]);
          if (citation) {
            pieces.push(
              <button
                key={`${li}-${index}`}
                className="cite-chip"
                onClick={() => onOpen(citation.article_id)}
                title={`${s.openArticle}: ${citation.law_name}`}
              >
                {citation.title}
              </button>,
            );
          }
          last = index + match[0].length;
        }
        pieces.push(line.slice(last));
        const bullet = /^\s*[-*]\s+/.test(line);
        return bullet ? (
          <li key={li}>{pieces.map((p) => (typeof p === "string" ? p.replace(/^\s*[-*]\s+/, "") : p))}</li>
        ) : (
          <p key={li}>{pieces}</p>
        );
      })}
    </div>
  );
}

function CitationCard({ c, uiLang, onOpen }: { c: Citation; uiLang: Lang; onOpen: (id: string) => void }) {
  const s = t(uiLang);
  return (
    <article className={`citation ${c.cited ? "is-cited" : ""}`}>
      <header>
        <span className="rank">{c.rank}</span>
        <button className="link-button citation-title" onClick={() => onOpen(c.article_id)}>
          {c.title}
        </button>
        <span className="law-name">{c.law_name}</span>
        {c.cited && <span className="badge badge-cited">{s.cited}</span>}
      </header>
      {c.passage && <Highlighted passage={c.passage} />}
      <footer>
        <button className="link-button" onClick={() => onOpen(c.article_id)}>
          {s.readArticle}
        </button>
        <a href={c.url} target="_blank" rel="noreferrer">
          {s.openAlmeezan}
        </a>
      </footer>
    </article>
  );
}

export function AnswerView({ answer, uiLang, onOpen }: Props) {
  const s = t(uiLang);
  const generative = answer.mode === "generative";
  const written = generative && !answer.refused;
  const badge = !generative ? s.extractiveBadge : answer.refused ? s.refusedBadge : s.generativeBadge;
  const label = !generative ? s.extractiveLabel : answer.refused ? s.refusedLabel : s.generativeLabel;
  const cited = answer.citations.filter((c) => c.cited);
  const others = answer.citations.filter((c) => !c.cited);
  return (
    <section className="answer" aria-live="polite">
      <div className="answer-head">
        <h2>{s.answer}</h2>
        <span className={`badge ${written ? "badge-gen" : answer.refused ? "badge-warn" : "badge-ext"}`}>{badge}</span>
        {answer.model && <span className="muted small">{answer.model}</span>}
      </div>
      <p className="mode-label small">{label}</p>
      {written && (
        <p className="answer-warning small" role="note">
          {s.answerWarning}
        </p>
      )}
      {written ? (
        <AnswerText answer={answer} uiLang={uiLang} onOpen={onOpen} />
      ) : (
        <p className="lead" dir={answer.language === "ar" ? "rtl" : "ltr"}>
          {answer.text}
        </p>
      )}
      {answer.notes.map((note) => (
        <p key={note} className="note small">
          {note}
        </p>
      ))}
      {answer.translated_question && (
        <p className="muted small">
          {s.searchedAs}: <span dir="auto">{answer.translated_question}</span>
        </p>
      )}

      {cited.length > 0 && (
        <>
          <h3>{s.cited}</h3>
          {cited.map((c) => (
            <CitationCard key={c.article_id} c={c} uiLang={uiLang} onOpen={onOpen} />
          ))}
        </>
      )}
      {others.length > 0 && (
        <>
          <h3>{cited.length ? s.alsoRetrieved : s.sources}</h3>
          {others.map((c) => (
            <CitationCard key={c.article_id} c={c} uiLang={uiLang} onOpen={onOpen} />
          ))}
        </>
      )}
    </section>
  );
}
