"""Mise en forme des réponses pour l'interface (sans dépendance à Streamlit, donc testable)."""

EXAMPLES = [
    "Quel est le délai pour retourner un article ?",
    "Combien coûte la livraison standard ?",
    "Dans quelles villes la livraison express est-elle disponible ?",
    "Comment annuler ma commande ?",
    "Livrez-vous à l'étranger ?",
    "Quelle est la capitale du Japon ?",
]

DISCLAIMER = (
    "Assistant automatique : les réponses proviennent uniquement de la documentation "
    "ShopFès et citent leurs sources. Pour toute décision (remboursement, litige), "
    "contactez le service client."
)

LATENCY_LABELS = {
    "translation": "traduction",
    "retrieval": "recherche",
    "generation": "génération",
    "retry_wait": "attente limite de débit",
}


def format_source(source: dict) -> str:
    """« [1] policies/return_policy.md — Return window · version 2026-09 · score 0.69 »."""
    parts = [f"[{source['ref']}] {source.get('source') or '?'} — {source.get('title') or ''}".rstrip(" —")]
    if source.get("version"):
        parts.append(f"version {source['version']}")
    if source.get("score") is not None:
        parts.append(f"score {source['score']:.2f}")
    return " · ".join(parts)


def format_latency(latency: dict) -> str:
    """« 1.8 s (traduction 0.4 s · recherche 0.02 s · génération 1.4 s) »."""
    total = latency.get("total")
    if total is None:
        return ""
    details = [f"{label} {latency[key] / 1000:.2f} s" for key, label in LATENCY_LABELS.items()
               if latency.get(key)]
    return f"{total / 1000:.1f} s" + (f" ({' · '.join(details)})" if details else "")


def response_meta(resp, question: str) -> dict:
    """Tout ce que l'interface affiche sous une réponse, calculé une seule fois."""
    translated = bool(resp.search_query) and resp.search_query.strip() != question.strip()
    return {
        "refused": resp.refused,
        "error": resp.error,
        "sources": [format_source(s) for s in resp.sources],
        "sources_title": ("Passages consultés (jugés insuffisants)" if resp.refused
                          else f"Sources ({len(resp.sources)})"),
        "search_query": resp.search_query if translated else None,
        "latency": format_latency(resp.latency_ms),
        "top_score": None if resp.top_score is None else round(resp.top_score, 2),
        "tokens": resp.prompt_tokens + resp.completion_tokens,
    }
