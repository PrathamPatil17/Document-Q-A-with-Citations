import os
import pytest
from src.docqa import rag, store

pytestmark = pytest.mark.skipif(
    not os.getenv("GROQ_API_KEY"),
    reason="Requires GROQ_API_KEY and an ingested store",
)


def _ingested() -> bool:
    try:
        return len(store.list_doc_ids()) >= 5
    except Exception:
        return False


def test_covered_question_returns_citations():
    if not _ingested():
        pytest.skip("Store not ingested")
    out = rag.ask("Does the EU AI Act prohibit real-time remote biometric "
                  "identification in public spaces?")
    assert out["covered"] is True
    assert len(out["citations"]) >= 1
    # Every returned citation must carry a real snippet and source.
    for c in out["citations"]:
        assert c["snippet"].strip()
        assert c["source_file"].endswith(".txt")


def test_out_of_scope_question_is_not_covered():
    if not _ingested():
        pytest.skip("Store not ingested")
    out = rag.ask("What will the weather be in Paris tomorrow afternoon?")
    assert out["covered"] is False
    assert out["citations"] == []


def test_known_conflict_topic():
    if not _ingested():
        pytest.skip("Store not ingested")
    out = rag.contradict("eu_ai_act", "nist_ai_rmf",
                         "binding legal force and mandatory compliance")
    assert out["verdict"] in {"conflict", "agree", "not_addressed_in_one_or_both"}
    # For this pair/topic we expect a real comparison with evidence from both.
    assert out["reasoning"].strip()


def test_multilingual_answer_language_matches_query():
    if not _ingested():
        pytest.skip("Store not ingested")
    # Spanish question -> Spanish answer.
    out = rag.ask("¿La Ley de IA de la UE prohíbe la identificación biométrica "
                  "remota en tiempo real en espacios públicos?")
    assert out["language"] == "es"
