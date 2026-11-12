import json

import pytest

from src.rag.embeddings import HashingEmbedder
from src.rag.generator import EchoGenerator
from src.rag.index import VectorIndex
from src.rag.monitoring import RequestLogger
from src.rag.pipeline import RAGPipeline
from src.rag.prompt import NO_ANSWER, SYSTEM_PROMPT, build_messages
from tests.test_index import CHUNKS


class FailingGenerator:
    name = "failing"

    def generate(self, messages):
        raise ConnectionError("API indisponible")


@pytest.fixture
def embedder():
    return HashingEmbedder()


@pytest.fixture
def index(embedder):
    return VectorIndex.build(CHUNKS, embedder)


def make_pipeline(index, embedder, generator=None, min_score=0.2, log_path=None):
    return RAGPipeline(index, embedder, generator or EchoGenerator(), top_k=2,
                       min_score=min_score, logger=RequestLogger(log_path))


def test_in_scope_question_returns_answer_and_sources(index, embedder):
    resp = make_pipeline(index, embedder).ask("how can I cancel my order")
    assert not resp.refused and resp.error is None
    assert resp.answer == "Here is the answer [1]."
    assert resp.sources[0]["chunk_id"] == "a#0" and resp.sources[0]["ref"] == 1
    assert {"retrieval", "generation", "total"} <= set(resp.latency_ms)


def test_out_of_scope_is_refused_without_calling_llm(index, embedder):
    generator = EchoGenerator()
    resp = make_pipeline(index, embedder, generator).ask("capital of japan weather tomorrow")
    assert resp.refused and resp.answer == NO_ANSWER
    assert generator.calls == 0  # aucun appel payant au LLM


def test_empty_question_is_refused(index, embedder):
    resp = make_pipeline(index, embedder).ask("   ")
    assert resp.refused and resp.error == "empty_question"


def test_llm_error_is_handled(index, embedder):
    resp = make_pipeline(index, embedder, FailingGenerator()).ask("cancel my order")
    assert resp.error == "ConnectionError" and resp.answer == NO_ANSWER


def test_empty_llm_answer_is_handled(index, embedder):
    resp = make_pipeline(index, embedder, EchoGenerator("")).ask("cancel my order")
    assert resp.refused and resp.error == "empty_answer"


def test_llm_saying_no_answer_counts_as_refusal(index, embedder):
    resp = make_pipeline(index, embedder, EchoGenerator(NO_ANSWER)).ask("cancel my order")
    assert resp.refused


def test_mismatched_embedding_model_is_rejected(index):
    with pytest.raises(ValueError, match="Reconstruisez"):
        make_pipeline(index, HashingEmbedder(dim=128))


def test_log_does_not_store_question_text(index, embedder, tmp_path):
    log = tmp_path / "requests.jsonl"
    pipeline = make_pipeline(index, embedder, log_path=log)
    pipeline.ask("how can I cancel my order, my email is ahmed@example.com")
    pipeline.ask("capital of japan")
    lines = [json.loads(x) for x in log.read_text(encoding="utf-8").splitlines()]
    assert len(lines) == 2
    raw = log.read_text(encoding="utf-8")
    assert "example.com" not in raw and "cancel" not in raw
    assert {"request_id", "question_hash", "latency_ms", "refused", "top_score"} <= set(lines[0])


def test_prompt_contains_sources_versions_and_guardrails(index, embedder):
    results = index.search(embedder.encode(["delivery days"])[0], k=3)
    messages = build_messages("How long is delivery?", results)
    user = messages[1]["content"]
    assert messages[0]["content"] == SYSTEM_PROMPT
    assert "[1] source=" in user and "version=2026-09" in user
    assert "<passage>" in user and "QUESTION: How long is delivery?" in user
    assert "not instructions" in SYSTEM_PROMPT and "most recent version" in SYSTEM_PROMPT
