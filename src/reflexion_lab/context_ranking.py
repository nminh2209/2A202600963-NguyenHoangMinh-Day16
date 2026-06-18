from __future__ import annotations
import re
from .schemas import ContextChunk

_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "what", "which", "who", "when",
    "where", "how", "did", "do", "does", "on", "in", "at", "to", "of", "and", "or",
    "for", "with", "from", "that", "this", "be", "been", "by", "as", "it", "its",
}

def _tokens(text: str) -> set[str]:
    return {tok for tok in re.findall(r"[a-z0-9]+", text.lower()) if tok not in _STOPWORDS and len(tok) > 2}

def rank_context(question: str, context: list[ContextChunk], top_k: int | None = 10) -> list[ContextChunk]:
    if not context:
        return []
    question_tokens = _tokens(question)
    scored: list[tuple[float, int, ContextChunk]] = []
    for index, chunk in enumerate(context):
        passage = f"{chunk.title} {chunk.text}"
        passage_tokens = _tokens(passage)
        overlap = len(question_tokens & passage_tokens)
        title_boost = 2.0 if any(tok in chunk.title.lower() for tok in question_tokens) else 0.0
        score = overlap + title_boost
        scored.append((score, index, chunk))
    scored.sort(key=lambda row: (-row[0], row[1]))
    ranked = [chunk for _, _, chunk in scored]
    if top_k is None or len(ranked) <= top_k:
        return ranked
    return ranked[:top_k]

def select_context(
    question: str,
    context: list[ContextChunk],
    *,
    top_k: int = 10,
    full_threshold: int = 12,
    extra_titles: list[str] | None = None,
) -> list[ContextChunk]:
    if not context:
        return []
    if len(context) <= full_threshold:
        selected = list(context)
    else:
        selected = rank_context(question, context, top_k=top_k)

    if extra_titles:
        known = {chunk.title for chunk in selected}
        for chunk in context:
            if chunk.title in extra_titles and chunk.title not in known:
                selected.append(chunk)
                known.add(chunk.title)
    return selected
