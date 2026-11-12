import pytest

from src.rag.embeddings import HashingEmbedder
from src.rag.generator import EchoGenerator, GenerationResult
from src.rag.index import VectorIndex
from src.rag.pipeline import RAGPipeline
from src.rag.translation import QueryTranslator
from tests.test_index import CHUNKS


class DictTranslatorBackend:
    """Faux générateur de traduction : renvoie une traduction fixe."""

    name = "fake-translator"

    def __init__(self, mapping: dict, tokens: int = 5):
        self.mapping, self.tokens, self.calls = mapping, tokens, 0

    def generate(self, messages):
        self.calls += 1
        question = messages[-1]["content"]
        return GenerationResult(self.mapping.get(question, question), self.tokens, self.tokens)


class FailingBackend:
    name = "failing"

    def generate(self, messages):
        raise TimeoutError("trop lent")


FR = "Comment annuler ma commande ?"
EN = "How do I cancel my order?"


def test_translation_success():
    result = QueryTranslator(DictTranslatorBackend({FR: EN})).translate(FR)
    assert result.text == EN and not result.used_fallback and result.error is None


def test_translation_strips_quotes():
    result = QueryTranslator(DictTranslatorBackend({FR: f'"{EN}"'})).translate(FR)
    assert result.text == EN


def test_translation_error_falls_back_to_original():
    result = QueryTranslator(FailingBackend()).translate(FR)
    assert result.text == FR and result.used_fallback and result.error == "TimeoutError"


@pytest.mark.parametrize("bad", ["", "x" * 500])
def test_empty_or_too_long_translation_is_rejected(bad):
    result = QueryTranslator(DictTranslatorBackend({FR: bad})).translate(FR)
    assert result.text == FR and result.used_fallback and result.error == "invalid_translation"


class RecordingGenerator(EchoGenerator):
    def generate(self, messages):
        self.last_user_message = messages[1]["content"]
        return super().generate(messages)


def test_pipeline_searches_with_translation_but_answers_original_question():
    embedder = HashingEmbedder()
    index = VectorIndex.build(CHUNKS, embedder)
    generator = RecordingGenerator()
    pipeline = RAGPipeline(index, embedder, generator, top_k=2, min_score=0.2,
                           translator=QueryTranslator(DictTranslatorBackend({FR: EN})))
    resp = pipeline.ask(FR)
    assert resp.search_query == EN and not resp.translation_fallback
    assert resp.sources[0]["chunk_id"] == "a#0"                  # trouvé grâce à la traduction
    assert f"QUESTION: {FR}" in generator.last_user_message      # le LLM reçoit la question originale
    assert "translation" in resp.latency_ms
    assert resp.prompt_tokens >= 5                       # tokens de traduction comptés


def test_refused_response_keeps_consulted_sources():
    embedder = HashingEmbedder()
    index = VectorIndex.build(CHUNKS, embedder)
    pipeline = RAGPipeline(index, embedder, EchoGenerator(), top_k=2, min_score=0.99)
    resp = pipeline.ask("cancel my order")
    assert resp.refused and len(resp.sources) == 2   # utile pour diagnostiquer un refus
