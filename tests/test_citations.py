from src.docqa.store import Retrieved
from src.docqa.rag import validate_citations


def _r(idx):
    return Retrieved("d1", "d1.txt", "Sec", idx, f"text {idx}", 0.2)


def test_keeps_only_real_cited_chunks():
    retrieved = [_r(0), _r(1), _r(2)]
    kept = validate_citations([1, 2], retrieved)
    assert [r.chunk_index for r in kept] == [1, 2]


def test_drops_fabricated_indexes():
    retrieved = [_r(0), _r(1)]
    kept = validate_citations([1, 99], retrieved)  # 99 was never retrieved
    assert [r.chunk_index for r in kept] == [1]


def test_empty_when_no_valid_citations():
    retrieved = [_r(0)]
    kept = validate_citations([42], retrieved)
    assert kept == []
