"""Métriques de retrieval.

Vérité terrain : chaque question Bitext a un `expected_intent`. Un chunk est
considéré pertinent s'il porte le même intent.

Limite à documenter : les chunks des politiques ShopFès n'ont pas d'intent
(intent=None). Une question sur les remboursements qui retrouve la politique de
retour est donc comptée comme un échec alors qu'elle peut être pertinente.
Recall@k est ainsi une borne basse de la vraie qualité de retrieval.
"""

import numpy as np


def first_relevant_rank(retrieved_intents: list, expected_intent: str) -> int | None:
    for rank, intent in enumerate(retrieved_intents, start=1):
        if intent == expected_intent:
            return rank
    return None


def recall_at_k(ranks: list[int | None], k: int) -> float:
    if not ranks:
        return 0.0
    return sum(r is not None and r <= k for r in ranks) / len(ranks)


def mean_reciprocal_rank(ranks: list[int | None]) -> float:
    if not ranks:
        return 0.0
    return sum(1.0 / r for r in ranks if r is not None) / len(ranks)


def best_threshold(in_scope_scores: list[float], out_scope_scores: list[float]) -> dict:
    """Seuil sur le meilleur score qui maximise l'exactitude équilibrée.

    Une question est refusée si son meilleur score < seuil. On veut refuser les
    questions hors périmètre (vrais refus) sans refuser celles du périmètre.
    """
    if not in_scope_scores or not out_scope_scores:
        return {"threshold": None, "balanced_accuracy": None}
    candidates = sorted(set(in_scope_scores) | set(out_scope_scores))
    best = {"threshold": None, "balanced_accuracy": -1.0}
    ins, outs = np.array(in_scope_scores), np.array(out_scope_scores)
    for t in candidates:
        accept_in = float(np.mean(ins >= t))   # questions du périmètre acceptées
        reject_out = float(np.mean(outs < t))  # questions hors périmètre refusées
        bal = (accept_in + reject_out) / 2
        if bal > best["balanced_accuracy"]:
            best = {"threshold": round(float(t), 4), "balanced_accuracy": round(bal, 4),
                    "in_scope_accepted": round(accept_in, 4),
                    "out_scope_rejected": round(reject_out, 4)}
    return best


def evaluate_retrieval(index, embedder, questions: list[dict], ks: list[int],
                       min_score: float) -> tuple[dict, list[dict]]:
    """Renvoie (métriques agrégées, détail par question pour l'analyse des échecs)."""
    max_k = max(ks)
    vectors = embedder.encode([q["question"] for q in questions])
    details, ranks, in_scores, out_scores = [], [], [], []

    for q, vec in zip(questions, vectors, strict=True):
        results = index.search(vec, max_k)
        top_score = results[0].score if results else float("-inf")
        retrieved = [r.chunk.get("intent") for r in results]
        row = {
            "qid": q["qid"],
            "in_scope": q["in_scope"],
            "expected_intent": q.get("expected_intent"),
            "top_score": round(top_score, 4),
            "retrieved_intents": retrieved,
            "retrieved_chunks": [r.chunk["chunk_id"] for r in results],
            "refused": top_score < min_score,
        }
        if q["in_scope"]:
            rank = first_relevant_rank(retrieved, q["expected_intent"])
            ranks.append(rank)
            in_scores.append(top_score)
            row["first_relevant_rank"] = rank
        else:
            out_scores.append(top_score)
        details.append(row)

    metrics = {f"recall_at_{k}": round(recall_at_k(ranks, k), 4) for k in ks}
    metrics["mrr"] = round(mean_reciprocal_rank(ranks), 4)
    metrics["n_in_scope"] = len(in_scores)
    metrics["n_out_of_scope"] = len(out_scores)
    if in_scores:
        metrics["in_scope_false_refusal_rate"] = round(float(np.mean(np.array(in_scores) < min_score)), 4)
        metrics["in_scope_top_score_mean"] = round(float(np.mean(in_scores)), 4)
    if out_scores:
        metrics["out_scope_refusal_rate"] = round(float(np.mean(np.array(out_scores) < min_score)), 4)
        metrics["out_scope_top_score_mean"] = round(float(np.mean(out_scores)), 4)
    suggestion = best_threshold(in_scores, out_scores)
    metrics["suggested_min_score"] = suggestion["threshold"]
    metrics["suggested_balanced_accuracy"] = suggestion["balanced_accuracy"]
    return metrics, details
