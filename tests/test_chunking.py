import pytest

from src.data.chunking import chunk_documents, split_words


def test_short_text_gives_one_chunk():
    assert split_words("a b c", chunk_size=10, chunk_overlap=2) == ["a b c"]


def test_empty_text_gives_no_chunk():
    assert split_words("   ", chunk_size=10, chunk_overlap=2) == []


def test_chunks_respect_size_and_overlap():
    words = [f"w{i}" for i in range(25)]
    chunks = split_words(" ".join(words), chunk_size=10, chunk_overlap=3)
    assert all(len(c.split()) <= 10 for c in chunks)
    # les 3 derniers mots d'un chunk ouvrent le suivant
    for a, b in zip(chunks, chunks[1:], strict=False):
        assert a.split()[-3:] == b.split()[:3]


def test_no_word_is_lost():
    words = [f"w{i}" for i in range(37)]
    chunks = split_words(" ".join(words), chunk_size=10, chunk_overlap=4)
    covered = {w for c in chunks for w in c.split()}
    assert covered == set(words)


@pytest.mark.parametrize("size,overlap", [(0, 0), (10, 10), (10, -1)])
def test_invalid_parameters_raise(size, overlap):
    with pytest.raises(ValueError):
        split_words("a b c", size, overlap)


def test_metadata_is_propagated():
    docs = [{"doc_id": "d1", "source": "bitext", "intent": "cancel_order", "text": "x " * 50}]
    chunks = chunk_documents(docs, chunk_size=20, chunk_overlap=5)
    assert len(chunks) > 1
    for i, c in enumerate(chunks):
        assert c["chunk_id"] == f"d1#{i}"
        assert c["source"] == "bitext" and c["intent"] == "cancel_order"
        assert c["n_chunks"] == len(chunks)


def test_chunk_ids_are_unique():
    docs = [{"doc_id": f"d{i}", "text": "x " * 100} for i in range(5)]
    chunks = chunk_documents(docs, chunk_size=30, chunk_overlap=10)
    ids = [c["chunk_id"] for c in chunks]
    assert len(ids) == len(set(ids))
