"""Indicateurs de monitoring calculés à partir du journal des requêtes.

Source : logs/requests.jsonl, écrit par RAGPipeline (une ligne par requête, sans
le texte de la question). Couvre les dimensions de la section 9 du cahier des
charges : disponibilité et erreurs, performance (P50/P95), qualité observable
(refus, scores de retrieval), coût et ressources (tokens), dérive des entrées.

Limite : la qualité réelle des réponses n'est pas observable en production sans
retour utilisateur ; on suit ici des indicateurs indirects (refus, score de
retrieval, échecs de traduction). La qualité mesurée vient de l'évaluation hors
ligne (reports/generation_metrics.json).
"""

from collections import defaultdict

import numpy as np


def _service(r: dict) -> float | None:
    """Latence de service, ou None si la requête date d'avant l'instrumentation.

    Les anciennes lignes n'ont pas de champ « service » : l'attente due aux limites de
    débit n'y était pas mesurée (la bibliothèque groq réessayait en silence). Leur
    latence totale mélange calcul et attente ; on ne l'assimile PAS à du service.
    """
    return (r.get("latency_ms") or {}).get("service")


def is_instrumented(r: dict) -> bool:
    return _service(r) is not None


def _pct(values: list[float], q: float) -> float | None:
    return round(float(np.percentile(values, q)), 1) if values else None


def _mean(values: list[float]) -> float | None:
    return round(float(np.mean(values)), 4) if values else None


def summarize(records: list[dict], price_in: float, price_out: float) -> dict:
    n = len(records)
    if n == 0:
        return {"n_requests": 0}
    instrumented = [r for r in records if is_instrumented(r)]
    service = [_service(r) for r in instrumented]
    total = [r["latency_ms"]["total"] for r in records
             if (r.get("latency_ms") or {}).get("total") is not None]
    tokens_in = [r.get("prompt_tokens", 0) or 0 for r in records]
    tokens_out = [r.get("completion_tokens", 0) or 0 for r in records]
    scores = [r["top_score"] for r in records if r.get("top_score") is not None]
    translated = [r for r in records if r.get("translated")]
    cost = sum(i * price_in + o * price_out for i, o in zip(tokens_in, tokens_out, strict=True)) / 1e6
    return {
        "n_requests": n,
        "first": records[0].get("timestamp"),
        "last": records[-1].get("timestamp"),
        "error_rate": round(sum(bool(r.get("error")) for r in records) / n, 4),
        "refusal_rate": round(sum(bool(r.get("refused")) for r in records) / n, 4),
        # Latence de service et limite de débit : seulement sur les requêtes instrumentées.
        "n_instrumented": len(instrumented),
        "latency_service_p50_ms": _pct(service, 50),
        "latency_service_p95_ms": _pct(service, 95),
        "latency_total_p95_ms": _pct(total, 95),
        "rate_limit_share": (round(sum((r["latency_ms"].get("retry_wait") or 0) > 0 for r in instrumented)
                                   / len(instrumented), 4) if instrumented else None),
        "translation_fallback_rate": (round(sum(bool(r.get("translation_fallback")) for r in translated)
                                            / len(translated), 4) if translated else None),
        "top_score_mean": _mean(scores),
        "question_length_mean": _mean([r.get("question_length", 0) for r in records]),
        "tokens_total": int(sum(tokens_in) + sum(tokens_out)),
        "tokens_per_request_mean": round((sum(tokens_in) + sum(tokens_out)) / n, 1),
        "cost_total_usd": round(cost, 6),
        "cost_per_1000_requests_usd": round(cost / n * 1000, 4),
    }


def by_hour(records: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for r in records:
        groups[(r.get("timestamp") or "")[:13]].append(r)  # « 2026-11-12T20 »
    rows = []
    for hour in sorted(groups):
        g = groups[hour]
        service = [v for v in (_service(r) for r in g) if v is not None]
        rows.append({
            "hour": hour.replace("T", " ") + "h" if hour else "inconnue",
            "requests": len(g),
            "errors": sum(bool(r.get("error")) for r in g),
            "refusals": sum(bool(r.get("refused")) for r in g),
            "latency_service_p95_ms": _pct(service, 95),
            "tokens": int(sum((r.get("prompt_tokens") or 0) + (r.get("completion_tokens") or 0) for r in g)),
        })
    return rows


def drift(records: list[dict], window: int) -> dict:
    """Compare les `window` requêtes les plus récentes aux `window` précédentes."""
    if len(records) < 2 * window:
        return {"available": False, "reason": f"au moins {2 * window} requêtes nécessaires ({len(records)})"}
    ref, cur = records[-2 * window:-window], records[-window:]

    def mean_score(rs):
        return _mean([r["top_score"] for r in rs if r.get("top_score") is not None])

    def mean_len(rs):
        return _mean([r.get("question_length", 0) for r in rs])

    ref_s, cur_s, ref_l, cur_l = mean_score(ref), mean_score(cur), mean_len(ref), mean_len(cur)
    return {
        "available": True,
        "window": window,
        "top_score_ref": ref_s,
        "top_score_current": cur_s,
        "top_score_drop": round(ref_s - cur_s, 4) if ref_s is not None and cur_s is not None else None,
        "question_length_ref": ref_l,
        "question_length_current": cur_l,
        "question_length_change": round(abs(cur_l - ref_l) / ref_l, 4) if ref_l else None,
    }


def alerts(summary: dict, drift_info: dict, rules: dict) -> list[dict]:
    """Chaque règle produit une alerte OK / ALERTE, avec la valeur et le seuil."""
    checks = [
        ("Taux d'erreur", summary.get("error_rate"), rules["error_rate_max"], "max"),
        ("Latence de service P95 (ms)", summary.get("latency_service_p95_ms"),
         rules["latency_service_p95_max_ms"], "max"),
        ("Requêtes ralenties par la limite de débit", summary.get("rate_limit_share"),
         rules["rate_limit_share_max"], "max"),
        ("Échecs de traduction", summary.get("translation_fallback_rate"),
         rules["translation_fallback_max"], "max"),
    ]
    if drift_info.get("available"):
        checks += [
            ("Baisse du score de retrieval (dérive)", drift_info.get("top_score_drop"),
             rules["top_score_drop_max"], "max"),
            ("Variation de la longueur des questions (dérive)", drift_info.get("question_length_change"),
             rules["question_length_change_max"], "max"),
        ]
    out = []
    for name, value, threshold, _ in checks:
        if value is None:
            out.append({"name": name, "status": "n/a", "value": None, "threshold": threshold})
            continue
        out.append({"name": name, "status": "ok" if value <= threshold else "alert",
                    "value": value, "threshold": threshold})
    return out
