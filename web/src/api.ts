// Types and calls for the qanun HTTP API (see src/qanun/api/schemas.py).

export type Lang = "en" | "ar";

export interface Passage {
  text: string;
  lang: Lang;
  highlights: [number, number][];
}

export interface Citation {
  article_id: string;
  law: string;
  label: string;
  title: string;
  law_name: string;
  url: string;
  rank: number;
  score: number;
  cited: boolean;
  passage: Passage | null;
}

export interface Answer {
  question: string;
  language: Lang;
  mode: "extractive" | "generative";
  text: string;
  refused: boolean;
  model: string | null;
  notes: string[];
  translated_question: string | null;
  citations: Citation[];
}

export interface SearchHit {
  rank: number;
  score: number;
  article_id: string;
  law: string;
  label: string;
  title_en: string;
  title_ar: string;
  passage: Passage | null;
}

export interface SearchResult {
  query: string;
  method: string;
  hits: SearchHit[];
}

export interface Amendment {
  instrument_ar: string;
  instrument_en: string | null;
  instrument_article: string | null;
  effective_date: string | null;
  change: "amended" | "added";
}

export interface Article {
  id: string;
  law: string;
  law_name_en: string;
  law_name_ar: string;
  label: string;
  number: number;
  kind: string;
  title_en: string;
  title_ar: string;
  chapter_en: string | null;
  chapter_ar: string | null;
  section_en: string | null;
  section_ar: string | null;
  text_ar: string;
  text_en: string | null;
  text_en_source: "almeezan" | "almeezan-legacy" | null;
  status: string;
  effective_date: string | null;
  source_url: string;
  amendments: Amendment[];
  history: { text_ar: string; effective_from: string | null; effective_to: string | null }[];
  notes: string[];
}

export interface Health {
  status: string;
  articles: number;
  search_methods: string[];
  default_method: string;
  embedding_model: string | null;
  reranker: string | null;
  generation: boolean;
  translation: boolean;
  llm: string | null;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const resp = await fetch(path, init);
  if (!resp.ok) {
    let detail = resp.statusText;
    try {
      const body = await resp.json();
      detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch {
      // keep the status text
    }
    throw new Error(`${resp.status}: ${detail}`);
  }
  return resp.json() as Promise<T>;
}

export const api = {
  health: () => request<Health>("/health"),
  ask: (question: string, k = 4) =>
    request<Answer>("/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, k }),
    }),
  search: (q: string, method: string, k = 10) =>
    request<SearchResult>(`/search?${new URLSearchParams({ q, method, k: String(k) })}`),
  article: (law: string, label: string) =>
    request<Article>(`/articles/${encodeURIComponent(label)}?law=${encodeURIComponent(law)}`),
};

/** "labour-2004:52-bis" -> { law: "labour-2004", label: "52-bis" } */
export function splitId(id: string): { law: string; label: string } {
  const i = id.indexOf(":");
  return { law: id.slice(0, i), label: id.slice(i + 1) };
}
