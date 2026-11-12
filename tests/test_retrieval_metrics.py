from src.evaluation.retrieval import (
    best_threshold,
    evaluate_retrieval,
    first_relevant_rank,
    mean_reciprocal_rank,
    recall_at_k,
)
from src.rag.embeddings import HashingEmbedder
from src.rag.index import VectorIndex
from tests.test_index import CHUNKS


def test_first_relevant_rank():
    assert first_relevant_rank(["a", "b", "c"], "b") == 2
    assert first_relevant_rank(["a", None], "z") is None


def test_recall_and_mrr():
    ranks = [1, 3, None, 2]
    assert recall_at_k(ranks, 1) == 0.25
    assert recall_at_k(ranks, 3) == 0.75
    assert abs(mean_reciprocal_rank(ranks) - (1 + 1 / 3 + 1 / 2) / 4) < 1e-9
    assert recall_at_k([], 3) == 0.0


def test_best_threshold_separates_clean_scores():
    result = best_threshold([0.8, 0.7, 0.75], [0.1, 0.2])
    assert result["balanced_accuracy"] == 1.0
    assert 0.2 < result["threshold"] <= 0.7


def test_best_threshold_without_data():
    assert best_threshold([], [0.1])["threshold"] is None


def test_evaluate_retrieval_end_to_end():
    embedder = HashingEmbedder()
    index = VectorIndex.build(CHUNKS, embedder)
    questions = [
        {"qid": "q1", "question": "cancel my order", "expected_intent": "cancel_order", "in_scope": True},
        {"qid": "q2", "question": "refund status", "expected_intent": "track_refund", "in_scope": True},
        {"qid": "q3", "question": "capital of japan", "expected_intent": None, "in_scope": False},
    ]
    metrics, details = evaluate_retrieval(index, embedder, questions, ks=[1, 3], min_score=0.2)
    assert metrics["recall_at_1"] == 1.0 and metrics["mrr"] == 1.0
    assert metrics["n_in_scope"] == 2 and metrics["n_out_of_scope"] == 1
    assert metrics["out_scope_refusal_rate"] == 1.0
    assert len(details) == 3 and details[0]["first_relevant_rank"] == 1
