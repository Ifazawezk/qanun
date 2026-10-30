import type { Passage } from "../api";

/** Passage text with the matched words wrapped in <mark>. */
export function Highlighted({ passage }: { passage: Passage }) {
  const parts: JSX.Element[] = [];
  let last = 0;
  passage.highlights.forEach(([start, end], i) => {
    if (start > last) parts.push(<span key={`t${i}`}>{passage.text.slice(last, start)}</span>);
    parts.push(<mark key={`m${i}`}>{passage.text.slice(start, end)}</mark>);
    last = end;
  });
  if (last < passage.text.length) parts.push(<span key="tail">{passage.text.slice(last)}</span>);
  return (
    <p className="passage" dir={passage.lang === "ar" ? "rtl" : "ltr"} lang={passage.lang}>
      {parts}
    </p>
  );
}
