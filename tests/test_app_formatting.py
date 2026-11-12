from src.app.formatting import EXAMPLES, format_latency, format_source, response_meta
from src.rag.pipeline import RAGResponse
from src.rag.prompt import NO_ANSWER

SRC = {"ref": 1, "chunk_id": "p#0", "source": "policies/return_policy.md", "title": "Return window",
       "version": "2026-09", "score": 0.6912}


def test_format_source_with_version_and_score():
    expected = "[1] policies/return_policy.md — Return window · version 2026-09 · score 0.69"
    assert format_source(SRC) == expected


def test_format_source_without_version():
    s = {"ref": 2, "source": "bitext", "title": "cancel order", "version": None, "score": 0.5}
    assert format_source(s) == "[2] bitext — cancel order · score 0.50"


def test_format_latency_skips_empty_steps():
    lat = {"translation": 400.0, "retrieval": 20.0, "generation": 1400.0, "retry_wait": 0.0,
           "service": 1820.0, "total": 1820.0}
    assert format_latency(lat) == "1.8 s (traduction 0.40 s · recherche 0.02 s · génération 1.40 s)"
    assert format_latency({}) == ""


def test_meta_for_translated_answer():
    resp = RAGResponse("r", "Vous avez 30 jours [1].", False, sources=[SRC], top_score=0.6912,
                       latency_ms={"total": 1000.0}, prompt_tokens=800, completion_tokens=50,
                       search_query="What is the return window?")
    meta = response_meta(resp, "Quel est le délai de retour ?")
    assert meta["sources_title"] == "Sources (1)" and meta["search_query"] == "What is the return window?"
    assert meta["top_score"] == 0.69 and meta["tokens"] == 850 and not meta["refused"]


def test_meta_for_refusal_hides_untranslated_query():
    resp = RAGResponse("r", NO_ANSWER, True, sources=[SRC], top_score=0.08, latency_ms={"total": 300.0},
                       search_query="capital of japan")
    meta = response_meta(resp, "capital of japan")
    assert meta["refused"] and meta["sources_title"].startswith("Passages consultés")
    assert meta["search_query"] is None  # même texte : rien à afficher


def test_examples_cover_answer_and_refusal():
    assert len(EXAMPLES) >= 5 and any("Japon" in e for e in EXAMPLES)
