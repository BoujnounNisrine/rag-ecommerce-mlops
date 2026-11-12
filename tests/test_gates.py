import copy

import pytest

from src.config import active_signature, load_params
from src.quality.gates import check_data, check_freshness, check_retrieval, load_gates

GOOD_METRICS = {
    "recall_at_1": 0.82, "recall_at_3": 0.94, "mrr": 0.88,
    "in_scope_false_refusal_rate": 0.0, "out_scope_refusal_rate": 0.9,
    "n_in_scope": 108, "n_out_of_scope": 30,
}


@pytest.fixture
def params():
    return load_params()


@pytest.fixture
def gates():
    return load_gates()


def make_report(params, embedder=None):
    sig = active_signature(params)
    return {"active": {"config": sig, "embedder_used": embedder or sig["embedding_model"],
                       "metrics": dict(GOOD_METRICS)}}


def failed(results):
    return {r.name for r in results if not r.passed}


def test_gates_file_has_justifications(gates):
    for name, rule in gates["retrieval"].items():
        assert ("min" in rule) != ("max" in rule), name
        assert rule.get("why"), f"Seuil {name} sans justification"


def test_good_metrics_pass(gates):
    assert failed(check_retrieval(GOOD_METRICS, gates["retrieval"])) == set()


@pytest.mark.parametrize("metric,value", [
    ("recall_at_3", 0.80), ("in_scope_false_refusal_rate", 0.10),
    ("out_scope_refusal_rate", 0.5), ("n_out_of_scope", 10),
])
def test_degraded_metric_fails(gates, metric, value):
    metrics = {**GOOD_METRICS, metric: value}
    assert failed(check_retrieval(metrics, gates["retrieval"])) == {metric}


def test_missing_metric_fails(gates):
    metrics = {k: v for k, v in GOOD_METRICS.items() if k != "mrr"}
    assert "mrr" in failed(check_retrieval(metrics, gates["retrieval"]))


def test_fresh_report_passes(params):
    assert failed(check_freshness(make_report(params), params)) == set()


def test_stale_report_fails(params):
    report = make_report(params)
    newer = copy.deepcopy(params)
    newer["chunking"]["chunk_size"] = 999
    results = check_freshness(report, newer)
    assert "report_matches_params" in failed(results)
    assert "chunk_size" in next(r.detail for r in results if r.name == "report_matches_params")


def test_fake_embedder_report_fails(params):
    assert "real_embedder_used" in failed(check_freshness(make_report(params, "hashing-256"), params))


def test_report_without_active_section_fails(params):
    assert failed(check_freshness({}, params)) == {"active_config_present"}


# Données -------------------------------------------------------------------

def make_data(n_docs=120):
    docs = [{"doc_id": f"d{i}", "source": "bitext", "doc_type": "faq_response",
             "intent": "cancel_order", "text": "cancel your order here"} for i in range(n_docs)]
    docs += [{"doc_id": f"p{i}", "source": "policies/x.md", "doc_type": "policy",
              "version": "2026-09", "text": "return within 30 days"} for i in range(6)]
    chunks = [{"chunk_id": f"{d['doc_id']}#0", "doc_id": d["doc_id"], "intent": d.get("intent"),
               "text": d["text"]} for d in docs]
    questions = [{"qid": "q1", "in_scope": True, "expected_intent": "cancel_order"},
                 {"qid": "o1", "in_scope": False, "expected_intent": None}]
    return docs, chunks, questions


def test_clean_data_passes(params, gates):
    assert failed(check_data(*make_data(), params, gates["data"])) == set()


def test_data_problems_are_detected(params, gates):
    docs, chunks, questions = make_data()
    docs[0]["doc_id"] = docs[1]["doc_id"]                       # doublon
    docs[-1]["version"] = None                                  # politique sans version
    chunks[2]["text"] = "call {{Order Number}}"                 # placeholder restant
    chunks[3]["doc_id"] = "ghost"                               # chunk orphelin
    chunks[4]["text"] = "word " * 10_000                        # chunk trop long
    questions.append({"qid": "q2", "in_scope": True, "expected_intent": "track_refund"})
    assert failed(check_data(docs, chunks, questions, params, gates["data"])) == {
        "unique_doc_ids", "policies_have_version", "no_placeholders",
        "chunk_provenance", "chunk_size_respected", "eval_intents_in_corpus",
    }


def test_too_few_documents_fails(params, gates):
    assert "min_documents" in failed(check_data(*make_data(n_docs=10), params, gates["data"]))
