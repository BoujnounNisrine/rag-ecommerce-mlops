import copy
import json

import pytest

from src.config import load_params, resolve
from src.data.io import read_jsonl
from src.evaluation.generation import (
    aggregate,
    check_answer,
    contains_term,
    evaluate,
    generation_signature,
    normalize,
    parse_judge,
)
from src.quality.gates import check_generation, load_gates
from src.rag.embeddings import HashingEmbedder
from src.rag.generator import EchoGenerator, GenerationResult, call_with_retry
from src.rag.index import VectorIndex
from src.rag.pipeline import RAGPipeline, RAGResponse
from src.rag.prompt import NO_ANSWER
from tests.test_index import CHUNKS

SOURCES = [
    {"ref": 1, "chunk_id": "p#0", "source": "policies/return_policy.md", "title": "Return window",
     "version": "2026-09", "score": 0.7},
    {"ref": 2, "chunk_id": "o#0", "source": "policies/warranty_old.md", "title": "Return window",
     "version": "2024-03", "score": 0.6},
]


def resp(answer, refused=False, sources=SOURCES, error=None):
    return RAGResponse("r1", answer, refused, sources=list(sources), latency_ms={"total": 100.0},
                       error=error)


ITEM = {"qid": "q", "expected": "answer", "must_include": [["30"]],
        "expected_sources": ["return_policy.md"]}


# Normalisation ---------------------------------------------------------------

def test_normalize_removes_accents_and_case():
    assert normalize("Fès  ÉTÉ") == "fes ete"


def test_numbers_match_whole_numbers_only():
    assert contains_term(normalize("sous 30 jours"), "30")
    assert not contains_term(normalize("sous 30 jours"), "3")
    assert contains_term(normalize("de 3 à 5 jours"), "3")


# Contrôles d'une réponse -----------------------------------------------------------

def test_good_answer_passes_all_checks():
    checks = check_answer(ITEM, resp("Vous avez 30 jours [1]."))
    assert checks == {"error": False, "behavior": True, "forbidden": True, "facts": True,
                      "citation": True, "citation_valid": True, "source": True}


def test_missing_fact_and_citation_are_detected():
    checks = check_answer(ITEM, resp("Vous avez 14 jours."))
    assert not checks["facts"] and not checks["citation"]
    assert "citation_valid" not in checks  # pas de citation : rien à valider, déjà compté par `citation`


def test_invented_citation_is_invalid():
    checks = check_answer(ITEM, resp("Vous avez 30 jours [1][7]."))
    assert checks["citation"] and not checks["citation_valid"]


def test_wrong_source_is_detected():
    checks = check_answer(ITEM, resp("Vous avez 30 jours [2]."))
    assert not checks["source"]


def test_refusal_on_answerable_question_fails_facts():
    checks = check_answer(ITEM, resp(NO_ANSWER, refused=True))
    assert not checks["behavior"] and not checks["facts"] and "citation" not in checks


def test_refuse_item():
    item = {"qid": "x", "expected": "refuse", "must_not_include": ["tokyo"]}
    assert check_answer(item, resp(NO_ANSWER, refused=True))["behavior"]
    bad = check_answer(item, resp("La capitale est Tokyo [1]."))
    assert not bad["behavior"] and not bad["forbidden"]


def test_clarify_accepts_refusal_or_question():
    item = {"qid": "a", "expected": "clarify"}
    assert check_answer(item, resp("Pouvez-vous préciser votre question ?"))["behavior"]
    assert check_answer(item, resp(NO_ANSWER, refused=True))["behavior"]
    assert not check_answer(item, resp("Trois jours [1]."))["behavior"]


def test_technical_error_is_flagged():
    assert check_answer(ITEM, resp(NO_ANSWER, refused=True, error="RateLimitError"))["error"]


# Juge -------------------------------------------------------------------------

def test_parse_judge_valid_and_invalid():
    ok = parse_judge('Voici : {"faithful": true, "score": 5, "unsupported_claims": []}')
    assert ok["faithful"] is True and ok["score"] == 5 and not ok["parse_error"]
    assert parse_judge("pas de json")["parse_error"]
    assert parse_judge('{"faithful": "yes", "score": 9}')["faithful"] is None


# Agrégation -------------------------------------------------------------------

def test_aggregate_rates_latency_and_cost():
    rows = [
        {"category": "factual", "expected": "answer",
         "checks": {"behavior": True, "facts": True, "error": False},
         "latency_ms": {"total": 100.0, "generation": 80.0}, "prompt_tokens": 1000, "completion_tokens": 100,
         "judge": {"faithful": True, "score": 5}},
        {"category": "injection", "expected": "refuse", "checks": {"behavior": False, "error": False},
         "latency_ms": {"total": 300.0}, "prompt_tokens": 0, "completion_tokens": 0},
    ]
    m = aggregate(rows, price_in=0.10, price_out=0.50)
    assert m["behavior_accuracy"] == 0.5 and m["refusal_accuracy"] == 0.0
    assert m["fact_accuracy"] == 1.0 and m["faithfulness_rate"] == 1.0
    assert m["latency_p95_ms"] == 290.0 and m["latency_p50_ms"] == 200.0
    assert m["cost_per_request_usd"] == pytest.approx((1000 * 0.10 + 100 * 0.50) / 1e6 / 2, abs=1e-6)
    assert set(m["by_category"]) == {"factual", "injection"}


# Bout en bout avec faux modèles ----------------------------------------------------

class FakeJudge:
    name = "fake-judge"

    def generate(self, messages):
        return GenerationResult('{"faithful": true, "score": 4, "unsupported_claims": []}', 10, 5)


def test_evaluate_end_to_end_with_fakes():
    embedder = HashingEmbedder()
    index = VectorIndex.build(CHUNKS, embedder)
    pipeline = RAGPipeline(index, embedder, EchoGenerator("Annulez depuis order history [1]."),
                           top_k=2, min_score=0.2)
    items = [
        {"qid": "a", "category": "bitext", "lang": "en", "question": "how to cancel my order",
         "expected": "answer", "must_include": [["order history"]], "expected_sources": ["bitext"]},
        {"qid": "b", "category": "off_topic", "lang": "en", "question": "capital of japan",
         "expected": "refuse"},
    ]
    rows = evaluate(pipeline, items, judge=FakeJudge())
    assert rows[0]["checks"]["facts"] and rows[0]["checks"]["source"] and rows[0]["judge"]["score"] == 4
    assert rows[1]["checks"]["behavior"] and "judge" not in rows[1]   # pas de juge sur un refus
    assert aggregate(rows, 0.1, 0.5)["behavior_accuracy"] == 1.0


# Jeu de questions versionné ------------------------------------------------------

def test_generation_question_file_is_valid():
    items = read_jsonl(resolve("data/eval/generation_questions.jsonl"))
    assert len(items) >= 25 and len({i["qid"] for i in items}) == len(items)
    for i in items:
        assert i["expected"] in {"answer", "refuse", "clarify"} and i["lang"] in {"fr", "en"}
    cats = {i["category"] for i in items}
    assert {"contradiction", "injection", "near_domain", "archived_only", "ambiguous"} <= cats


# Retry ------------------------------------------------------------------------

class RateLimitError(Exception):
    pass


def test_retry_on_rate_limit_then_success():
    calls, waits = [], []

    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise RateLimitError()
        return "ok"

    assert call_with_retry(flaky, sleep=waits.append) == "ok"
    assert waits == [2.0, 4.0]


def test_retry_stats_are_recorded():
    calls, stats = [], {}

    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise RateLimitError()
        return "ok"

    call_with_retry(flaky, sleep=lambda s: None, stats=stats)
    assert stats == {"retries": 2, "retry_wait_ms": 6000.0}


class SlowEchoGenerator(EchoGenerator):
    """Simule une réponse obtenue après 4 s d'attente imposée par la limite de débit."""

    def generate(self, messages):
        result = super().generate(messages)
        result.retry_wait_ms = 4000.0
        return result


def test_pipeline_separates_rate_limit_wait_from_service_latency():
    embedder = HashingEmbedder()
    index = VectorIndex.build(CHUNKS, embedder)
    pipeline = RAGPipeline(index, embedder, SlowEchoGenerator(), top_k=2, min_score=0.2)
    lat = pipeline.ask("cancel my order").latency_ms
    assert lat["retry_wait"] == 4000.0
    assert lat["service"] == pytest.approx(max(lat["total"] - 4000.0, 0.0), abs=0.2)


def test_aggregate_reports_service_latency():
    rows = [{"category": "c", "expected": "answer", "checks": {"behavior": True},
             "latency_ms": {"total": 9000.0, "retry_wait": 6000.0, "service": 3000.0},
             "prompt_tokens": 0, "completion_tokens": 0},
            {"category": "c", "expected": "answer", "checks": {"behavior": True},
             "latency_ms": {"total": 2000.0, "retry_wait": 0.0, "service": 2000.0},
             "prompt_tokens": 0, "completion_tokens": 0}]
    m = aggregate(rows, 0.1, 0.5)
    assert m["latency_p95_ms"] > 8000 and m["latency_service_p95_ms"] < 3000
    assert m["requests_with_rate_limit_wait"] == 1 and m["rate_limit_wait_total_s"] == 6.0


def test_no_retry_on_other_errors():
    calls = []

    def broken():
        calls.append(1)
        raise ValueError("clé invalide")

    with pytest.raises(ValueError):
        call_with_retry(broken, sleep=lambda s: None)
    assert len(calls) == 1


# Gates de génération -------------------------------------------------------------

GOOD = {"behavior_accuracy": 0.9, "fact_accuracy": 0.85, "citation_validity_rate": 1.0,
        "citation_rate": 0.95, "faithfulness_rate": 0.9, "forbidden_content_rate": 0.0,
        "error_rate": 0.0, "latency_service_p95_ms": 3000}


def gen_report(params, **overrides):
    return {"signature": generation_signature(params), "complete_run": True, "judge_used": True,
            "metrics": dict(GOOD), **overrides}


def failed(results):
    return {r.name for r in results if not r.passed}


def test_generation_gates_pass_and_detect_problems():
    params, gates = load_params(), load_gates()["generation"]
    assert failed(check_generation(gen_report(params), params, gates)) == set()
    leaked = gen_report(params, metrics={**GOOD, "forbidden_content_rate": 0.04})
    assert failed(check_generation(leaked, params, gates)) == {"forbidden_content_rate"}
    partial = gen_report(params, complete_run=False, judge_used=False)
    assert failed(check_generation(partial, params, gates)) == {"complete_run", "judge_used"}
    assert failed(check_generation(None, params, gates)) == {"report_exists"}


def test_changing_the_prompt_makes_generation_report_stale(monkeypatch):
    params = load_params()
    report = gen_report(params)
    monkeypatch.setattr("src.evaluation.generation.SYSTEM_PROMPT", "un autre prompt")
    assert "report_matches_params" in failed(check_generation(report, params, load_gates()["generation"]))


def test_changing_generation_params_makes_report_stale():
    params = load_params()
    report = gen_report(params)
    newer = copy.deepcopy(params)
    newer["generation"]["temperature"] = 0.7
    result = check_generation(report, newer, load_gates()["generation"])
    assert "report_matches_params" in failed(result)
    assert json.dumps([r.detail for r in result])  # détails sérialisables
