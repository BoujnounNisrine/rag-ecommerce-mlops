import numpy as np
import pytest

from src.rag.embeddings import HashingEmbedder
from src.rag.index import VectorIndex

CHUNKS = [
    {"chunk_id": "a#0", "intent": "cancel_order", "source": "bitext",
     "text": "To cancel your order go to order history and click cancel order"},
    {"chunk_id": "b#0", "intent": "track_refund", "source": "bitext",
     "text": "Your refund status is visible in the refund tracking page"},
    {"chunk_id": "c#0", "intent": None, "source": "policies/shipping_policy.md", "version": "2026-09",
     "text": "Standard delivery takes 3 to 5 business days in Morocco"},
]


@pytest.fixture
def embedder():
    return HashingEmbedder()


@pytest.fixture
def index(embedder):
    return VectorIndex.build(CHUNKS, embedder)


def test_embeddings_are_normalized(embedder):
    vectors = embedder.encode(["hello world", "another text", ""])
    norms = np.linalg.norm(vectors[:2], axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)
    assert np.allclose(vectors[2], 0.0)  # texte vide -> vecteur nul, pas de NaN


def test_embedder_is_deterministic(embedder):
    a = embedder.encode(["cancel my order"])
    b = HashingEmbedder().encode(["cancel my order"])
    assert np.array_equal(a, b)


def test_search_returns_most_similar_first(index, embedder):
    results = index.search(embedder.encode(["how do I cancel my order"])[0], k=2)
    assert results[0].chunk["chunk_id"] == "a#0"
    assert results[0].rank == 1 and results[1].rank == 2
    assert results[0].score >= results[1].score


def test_k_larger_than_index_is_capped(index, embedder):
    results = index.search(embedder.encode(["delivery"])[0], k=50)
    assert len(results) == len(CHUNKS)


def test_save_and_load_roundtrip(index, embedder, tmp_path):
    index.save(tmp_path)
    loaded = VectorIndex.load(tmp_path)
    assert loaded.meta == index.meta
    assert loaded.chunks == index.chunks  # provenance conservée
    q = embedder.encode(["refund status"])[0]
    assert [r.chunk["chunk_id"] for r in loaded.search(q, 3)] == [
        r.chunk["chunk_id"] for r in index.search(q, 3)
    ]


def test_meta_records_model_and_fingerprint(index):
    assert index.meta["embedding_model"] == "hashing-256"
    assert index.meta["n_chunks"] == 3 and len(index.meta["chunks_fingerprint"]) == 16


def test_missing_index_gives_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="dvc repro"):
        VectorIndex.load(tmp_path / "nope")


def test_empty_chunks_are_rejected(embedder):
    with pytest.raises(ValueError):
        VectorIndex.build([], embedder)
