from src.config import load_params
from src.monitoring.dashboard import render
from src.monitoring.metrics import alerts, by_hour, drift, summarize


def rec(i, service=1000.0, wait=0.0, score=0.6, refused=False, error=None, length=40, tokens=(800, 100),
        fallback=False, hour=20):
    return {"timestamp": f"2026-11-12T{hour}:{i % 60:02d}:00+00:00", "request_id": f"r{i}",
            "question_length": length, "refused": refused, "top_score": score, "error": error,
            "latency_ms": {"service": service, "retry_wait": wait, "total": service + wait},
            "prompt_tokens": tokens[0], "completion_tokens": tokens[1],
            "translated": True, "translation_fallback": fallback}


RULES = load_params()["monitoring"]["alerts"]


def by_name(alert_list):
    return {a["name"]: a for a in alert_list}


def test_summary_core_indicators():
    records = [rec(i) for i in range(18)] + [rec(18, error="RateLimitError", refused=True),
                                             rec(19, service=9000.0, wait=6000.0)]
    s = summarize(records, price_in=0.10, price_out=0.50)
    assert s["n_requests"] == 20 and s["error_rate"] == 0.05 and s["refusal_rate"] == 0.05
    assert s["rate_limit_share"] == 0.05
    assert s["latency_total_p95_ms"] > s["latency_service_p95_ms"]
    assert s["cost_per_1000_requests_usd"] == round((800 * 0.10 + 100 * 0.50) / 1e6 * 1000, 4)


OLD = {"timestamp": "2026-11-12T18:00:00+00:00", "refused": False, "top_score": 0.5,
       "latency_ms": {"retrieval": 20, "total": 12000.0}, "prompt_tokens": 0, "completion_tokens": 0}


def test_old_log_lines_are_not_counted_as_service_latency():
    """Avant l'instrumentation, la latence totale mélange calcul et attente : on ne
    l'assimile pas à de la latence de service (elle reste dans la latence totale)."""
    s = summarize([OLD, OLD], 0.1, 0.5)
    assert s["n_instrumented"] == 0 and s["latency_service_p95_ms"] is None
    assert s["rate_limit_share"] is None and s["latency_total_p95_ms"] == 12000.0
    assert s["translation_fallback_rate"] is None


def test_mixed_log_uses_only_instrumented_requests_for_service_latency():
    records = [OLD] * 90 + [rec(i, service=2000.0, wait=6000.0 if i % 2 else 0.0) for i in range(30)]
    s = summarize(records, 0.1, 0.5)
    assert s["n_instrumented"] == 30 and s["latency_service_p95_ms"] == 2000.0
    assert s["rate_limit_share"] == 0.5          # 15 / 30 instrumentées, pas 15 / 120
    a = by_name(alerts(s, drift(records, 40), RULES))
    assert a["Latence de service P95 (ms)"]["status"] == "ok"
    params = load_params()
    html = render(s, by_hour(records), drift(records, 40), list(a.values()), records, params)
    assert "avant instrumentation" in html and "service non mesuré" in html


def test_empty_log():
    assert summarize([], 0.1, 0.5) == {"n_requests": 0}


def test_hourly_breakdown():
    rows = by_hour([rec(1, hour=19), rec(2, hour=20), rec(3, hour=20, error="X")])
    assert [(r["hour"], r["requests"], r["errors"]) for r in rows] == [
        ("2026-11-12 19h", 1, 0), ("2026-11-12 20h", 2, 1)]


def test_drift_detects_score_drop_and_length_change():
    records = [rec(i, score=0.65, length=40) for i in range(10)] + \
              [rec(i, score=0.40, length=90) for i in range(10, 20)]
    d = drift(records, window=10)
    assert d["available"] and d["top_score_drop"] == 0.25 and d["question_length_change"] == 1.25
    a = by_name(alerts(summarize(records, 0.1, 0.5), d, RULES))
    assert a["Baisse du score de retrieval (dérive)"]["status"] == "alert"
    assert a["Variation de la longueur des questions (dérive)"]["status"] == "alert"


def test_drift_needs_enough_requests():
    assert not drift([rec(i) for i in range(5)], window=10)["available"]


def test_healthy_traffic_has_no_alert():
    records = [rec(i) for i in range(100)]
    a = alerts(summarize(records, 0.1, 0.5), drift(records, 40), RULES)
    assert {x["status"] for x in a} == {"ok"}


def test_slow_service_and_rate_limit_raise_alerts():
    records = [rec(i, service=12000.0, wait=2000.0) for i in range(20)]
    a = by_name(alerts(summarize(records, 0.1, 0.5), drift(records, 40), RULES))
    assert a["Latence de service P95 (ms)"]["status"] == "alert"
    assert a["Requêtes ralenties par la limite de débit"]["status"] == "alert"


def test_dashboard_html_is_self_contained_and_escaped():
    params = load_params()
    records = [rec(i) for i in range(30)] + [rec(30, error="<script>alert(1)</script>")]
    s = summarize(records, 0.1, 0.5)
    d = drift(records, 10)
    html = render(s, by_hour(records), d, alerts(s, d, RULES), records, params)
    assert html.startswith("<!doctype html>") and "<svg" in html
    assert "<script>alert(1)</script>" not in html            # texte échappé
    assert "http://" not in html and "https://" not in html   # aucune ressource externe
    assert "prefers-color-scheme:dark" in html                 # mode sombre
