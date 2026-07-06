"""RAG orchestration: /ask and /contradict pipelines, plus citation validation.

Guarding against hallucination happens in three places:
  1. Retrieval gate — if no chunk is within DISTANCE_THRESHOLD, we never call
     the LLM to answer and return covered=false.
  2. Prompt lock — the model is told to use only the provided context, cite the
     chunk indexes it used, and set "answerable": false if the context does not
     actually answer the question.
  3. Citation validation — any chunk index the model cites that was not
     retrieved is discarded; if nothing valid remains, we downgrade to
     covered=false.
"""
from .config import DISTANCE_THRESHOLD, TOP_K_ASK, TOP_K_CONTRADICT, DOC_BY_ID
from .embeddings import embed_query
from . import store, llm

_ASK_SYSTEM = (
    "You answer questions strictly from the provided policy-document excerpts. "
    "Rules: (1) Use ONLY the context. Do not use outside knowledge. "
    "(2) If the context does not actually answer the question, set "
    "\"answerable\" to false and leave the answer empty. "
    "(3) When you do answer, write a complete, self-contained answer of 2-4 "
    "sentences that states the conclusion and the key reasoning or conditions "
    "from the excerpts (do not reply with a bare 'Yes'/'No'). "
    "(4) Cite the chunk_index values you actually used. "
    "Respond as JSON: {\"answerable\": bool, \"answer\": str, "
    "\"cited_chunks\": [int]}."
)

_CONTRADICT_SYSTEM = (
    "You compare two policy documents on a topic and judge their relationship. "
    "Use ONLY the provided excerpts. Classify as exactly one of: "
    "\"conflict\" (they make incompatible claims/requirements), "
    "\"agree\" (they are consistent), or "
    "\"not_addressed_in_one_or_both\" (at least one does not address the topic). "
    "Quote the specific phrases that justify your verdict. "
    "Respond as JSON: {\"verdict\": str, \"reasoning\": str}."
)


def validate_citations(cited_indexes, retrieved):
    """Keep only retrieved chunks the model actually cited (drop fabrications)."""
    wanted = set(cited_indexes)
    return [r for r in retrieved if r.chunk_index in wanted]


def _format_context(retrieved) -> str:
    return "\n\n".join(
        f"[chunk_index={r.chunk_index} | {r.source_file} | {r.section_title}]\n{r.text}"
        for r in retrieved
    )


def _citation_dict(r) -> dict:
    return {
        "doc_id": r.doc_id,
        "source_file": r.source_file,
        "section": r.section_title,
        "chunk_index": r.chunk_index,
        "snippet": r.text,
    }


def _not_covered(language: str) -> dict:
    msg = "The provided documents do not address this question."
    if language != "en":
        try:
            msg = llm.translate(msg, language)
        except llm.LLMError:
            pass  # Fall back to English message rather than failing the request.
    return {"answer": msg, "covered": False, "citations": [], "language": language}


def ask(question: str) -> dict:
    language = llm.detect_language(question)
    # Translate non-English queries to English so retrieval matches the
    # English-only corpus. English queries skip the round trip.
    english_q = question if language == "en" else llm.translate(question, "en")

    retrieved = store.query(embed_query(english_q), TOP_K_ASK)
    relevant = [r for r in retrieved if r.distance <= DISTANCE_THRESHOLD]
    if not relevant:
        return _not_covered(language)

    result = llm.complete_json(
        _ASK_SYSTEM,
        f"Question: {english_q}\n\nContext:\n{_format_context(relevant)}",
    )
    if not result.get("answerable"):
        return _not_covered(language)

    cited = validate_citations(result.get("cited_chunks", []), relevant)
    if not cited:
        # Model claimed an answer but cited nothing valid -> treat as uncovered.
        return _not_covered(language)

    answer = result.get("answer", "").strip()
    if language != "en" and answer:
        answer = llm.translate(answer, language)

    return {
        "answer": answer,
        "covered": True,
        "citations": [_citation_dict(r) for r in cited],
        "language": language,
    }


def contradict(doc_id_1: str, doc_id_2: str, topic: str) -> dict:
    if doc_id_1 not in DOC_BY_ID or doc_id_2 not in DOC_BY_ID:
        raise ValueError("Unknown doc_id")

    q = embed_query(topic)
    r1 = [r for r in store.query(q, TOP_K_CONTRADICT, doc_id=doc_id_1)
          if r.distance <= DISTANCE_THRESHOLD]
    r2 = [r for r in store.query(q, TOP_K_CONTRADICT, doc_id=doc_id_2)
          if r.distance <= DISTANCE_THRESHOLD]

    if not r1 or not r2:
        return {
            "verdict": "not_addressed_in_one_or_both",
            "reasoning": "At least one document does not address this topic in "
                         "the ingested text.",
            "doc_1_evidence": [_citation_dict(r) for r in r1],
            "doc_2_evidence": [_citation_dict(r) for r in r2],
        }

    user = (
        f"Topic: {topic}\n\n"
        f"Document 1 ({doc_id_1}) excerpts:\n{_format_context(r1)}\n\n"
        f"Document 2 ({doc_id_2}) excerpts:\n{_format_context(r2)}"
    )
    result = llm.complete_json(_CONTRADICT_SYSTEM, user)
    return {
        "verdict": result.get("verdict", "not_addressed_in_one_or_both"),
        "reasoning": result.get("reasoning", ""),
        "doc_1_evidence": [_citation_dict(r) for r in r1],
        "doc_2_evidence": [_citation_dict(r) for r in r2],
    }
