"""Évaluation de la génération : qualité des réponses du système complet.

Chaque question annotée (data/eval/generation_questions.jsonl) passe par le
pipeline réel : traduction, retrieval, refus éventuel, génération Groq.

Contrôles automatiques par question :
  behavior   — répondre / refuser / demander une précision, comme attendu
  facts      — la réponse contient les faits attendus (`must_include`)
  forbidden  — la réponse ne contient pas ce qui est interdit (`must_not_include`)
  citation   — la réponse cite au moins une source [n]
  citation_valid — chaque [n] cité correspond à un passage réellement fourni
  source     — au moins une source citée provient du document attendu
  faithful   — (LLM juge) chaque affirmation est appuyée par les passages

Agrégats : taux par contrôle, par catégorie, latence moyenne / P50 / P95,
tokens et coût estimé par requête.

Usage :
  python -m src.evaluation.generation
  python -m src.evaluation.generation --no-judge --limit 5   (essai rapide)
"""

import argparse
import hashlib
import json
import re
import time
import unicodedata

import numpy as np

from src.config import load_params, resolve
from src.data.io import read_jsonl, write_jsonl
from src.rag.prompt import PROMPT_VERSION, SYSTEM_PROMPT

CITATION_RE = re.compile(r"\[(\d+)\]")

JUDGE_PROMPT = """You are a strict evaluator of a customer-support assistant.
You receive numbered PASSAGES and the assistant's ANSWER (possibly in French).
Decide whether EVERY factual claim in the answer is supported by the passages.
Ignore politeness, formatting and translation differences.
Reply with ONLY a JSON object, no other text:
{"faithful": true or false, "score": integer from 1 (invented) to 5 (fully supported),
 "unsupported_claims": [short list]}"""


# Utilitaires ---------------------------------------------------------------

def normalize(text: str) -> str:
    """Minuscules, sans accents, espaces normalisés (« Fès » == « fes »)."""
    text = unicodedata.normalize("NFKD", text or "")
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.replace(" ", " ").replace(" ", " ")
    return re.sub(r"\s+", " ", text.lower()).strip()


def contains_term(text_norm: str, term: str) -> bool:
    """Un nombre doit apparaître comme nombre entier (« 3 » ne matche pas « 30 »)."""
    term_norm = normalize(term)
    if term_norm.isdigit():
        return re.search(rf"(?<!\d){re.escape(term_norm)}(?!\d)", text_norm) is not None
    return term_norm in text_norm


def percentile(values: list[float], q: float) -> float | None:
    return round(float(np.percentile(values, q)), 1) if values else None


def file_sha(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def prompt_hash() -> str:
    return hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest()[:12]


def generation_signature(params: dict) -> dict:
    """Ce qui détermine les résultats de génération ; comparé par les quality gates."""
    r, g = params["retrieval"], params["generation"]
    return {
        "prompt_version": PROMPT_VERSION,
        "prompt_hash": prompt_hash(),
        "questions_sha": file_sha(resolve(params["data"]["generation_questions"])),
        "embedding_model": params["embeddings"]["model"],
        "chunk_size": params["chunking"]["chunk_size"],
        "top_k": r["top_k"],
        "min_score": r["min_score"],
        "translate_query": r.get("translate_query", False),
        "temperature": g["temperature"],
        "max_tokens": g["max_tokens"],
        "reasoning_effort": g.get("reasoning_effort"),
    }


# Contrôles -----------------------------------------------------------------

def check_answer(item: dict, resp) -> dict:
    """Contrôles automatiques d'une réponse. `resp` est un RAGResponse."""
    answer_norm = normalize(resp.answer)
    expected = item["expected"]
    checks = {"error": resp.error is not None and resp.error != "empty_question"}

    if expected == "answer":
        checks["behavior"] = not resp.refused
    elif expected == "refuse":
        checks["behavior"] = resp.refused
    else:  # clarify : refuser ou poser une question de précision
        checks["behavior"] = resp.refused or "?" in resp.answer

    checks["forbidden"] = not any(contains_term(answer_norm, t) for t in item.get("must_not_include", []))

    if expected == "answer" and resp.refused:
        checks["facts"] = False  # exactitude de bout en bout : un refus à tort est un échec
    if expected == "answer" and not resp.refused:
        groups = item.get("must_include", [])
        checks["facts"] = all(any(contains_term(answer_norm, t) for t in g) for g in groups)
        cited = [int(n) for n in CITATION_RE.findall(resp.answer)]
        refs = {s["ref"]: s for s in resp.sources}
        checks["citation"] = bool(cited)
        if cited:
            # Mesurée seulement sur les réponses qui citent : détecte les références
            # inventées ([7] alors que 4 passages). L'absence de citation est déjà
            # comptée par `citation`.
            checks["citation_valid"] = all(n in refs for n in cited)
        wanted = item.get("expected_sources", [])
        if wanted:
            cited_sources = [refs[n]["source"] or "" for n in cited if n in refs]
            checks["source"] = any(w in s for w in wanted for s in cited_sources)
    return checks


def parse_judge(text: str) -> dict:
    match = re.search(r"\{.*\}", text or "", flags=re.DOTALL)
    if not match:
        return {"faithful": None, "score": None, "unsupported_claims": [], "parse_error": True}
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {"faithful": None, "score": None, "unsupported_claims": [], "parse_error": True}
    score = data.get("score")
    return {
        "faithful": data.get("faithful") if isinstance(data.get("faithful"), bool) else None,
        "score": score if isinstance(score, int) and 1 <= score <= 5 else None,
        "unsupported_claims": data.get("unsupported_claims") or [],
        "parse_error": False,
    }


def judge_faithfulness(judge, answer: str, passages: list[str]) -> dict:
    context = "\n\n".join(f"[{i}] {p}" for i, p in enumerate(passages, start=1))
    messages = [
        {"role": "system", "content": JUDGE_PROMPT},
        {"role": "user", "content": f"PASSAGES:\n{context}\n\nANSWER:\n{answer}"},
    ]
    try:
        result = judge.generate(messages)
    except Exception as exc:
        return {"faithful": None, "score": None, "unsupported_claims": [],
                "parse_error": True, "error": type(exc).__name__}
    return {**parse_judge(result.text), "judge_tokens": result.prompt_tokens + result.completion_tokens}


# Agrégation ----------------------------------------------------------------

def rate(rows: list[dict], key: str) -> float | None:
    values = [r["checks"][key] for r in rows if key in r["checks"]]
    return round(sum(values) / len(values), 4) if values else None


def aggregate(rows: list[dict], price_in: float, price_out: float) -> dict:
    answerable = [r for r in rows if r["expected"] == "answer"]
    must_refuse = [r for r in rows if r["expected"] == "refuse"]
    latencies = [r["latency_ms"]["total"] for r in rows if r["latency_ms"].get("total") is not None]
    gen_latencies = [r["latency_ms"]["generation"] for r in rows if "generation" in r["latency_ms"]]
    service = [r["latency_ms"].get("service", r["latency_ms"].get("total")) for r in rows
               if r["latency_ms"].get("total") is not None]
    waits = [r["latency_ms"].get("retry_wait", 0.0) for r in rows]
    judged = [r["judge"] for r in rows if r.get("judge") and r["judge"].get("faithful") is not None]
    scores = [j["score"] for j in judged if j.get("score") is not None]
    tokens_in = [r["prompt_tokens"] for r in rows]
    tokens_out = [r["completion_tokens"] for r in rows]
    cost = [(i * price_in + o * price_out) / 1e6 for i, o in zip(tokens_in, tokens_out, strict=True)]

    metrics = {
        "n_questions": len(rows),
        "behavior_accuracy": rate(rows, "behavior"),
        "answer_rate_on_answerable": rate(answerable, "behavior"),
        "refusal_accuracy": rate(must_refuse, "behavior"),
        "fact_accuracy": rate(rows, "facts"),
        "citation_rate": rate(rows, "citation"),
        "citation_validity_rate": rate(rows, "citation_valid"),
        "source_accuracy": rate(rows, "source"),
        "forbidden_content_rate": round(1 - (rate(rows, "forbidden") or 1.0), 4),
        "error_rate": rate(rows, "error"),
        "faithfulness_rate": round(sum(j["faithful"] for j in judged) / len(judged), 4) if judged else None,
        "judge_mean_score": round(float(np.mean(scores)), 2) if scores else None,
        "n_judged": len(judged),
        "latency_mean_ms": round(float(np.mean(latencies)), 1) if latencies else None,
        "latency_p50_ms": percentile(latencies, 50),
        "latency_p95_ms": percentile(latencies, 95),
        "generation_latency_p95_ms": percentile(gen_latencies, 95),
        # Latence de service = latence totale moins l'attente imposée par les limites
        # de débit (palier gratuit Groq, appels rapprochés de l'évaluation).
        "latency_service_p50_ms": percentile(service, 50),
        "latency_service_p95_ms": percentile(service, 95),
        "requests_with_rate_limit_wait": sum(w > 0 for w in waits),
        "rate_limit_wait_total_s": round(sum(waits) / 1000, 1),
        "tokens_per_request_mean": round(float(np.mean(np.add(tokens_in, tokens_out))), 1) if rows else None,
        "cost_per_request_usd": round(float(np.mean(cost)), 6) if cost else None,
        "cost_per_1000_requests_usd": round(float(np.mean(cost)) * 1000, 4) if cost else None,
    }
    by_cat = {}
    for cat in sorted({r["category"] for r in rows}):
        sub = [r for r in rows if r["category"] == cat]
        by_cat[cat] = {"n": len(sub), "behavior": rate(sub, "behavior"), "facts": rate(sub, "facts")}
    metrics["by_category"] = by_cat
    return metrics


# Exécution -----------------------------------------------------------------

def evaluate(pipeline, items: list[dict], judge=None, sleep_seconds: float = 0.0) -> list[dict]:
    texts = {c["chunk_id"]: c["text"] for c in pipeline.index.chunks}
    rows = []
    for n, item in enumerate(items, start=1):
        resp = pipeline.ask(item["question"])
        row = {
            "qid": item["qid"], "category": item["category"], "lang": item["lang"],
            "expected": item["expected"], "question": item["question"],
            "answer": resp.answer, "refused": resp.refused, "error": resp.error,
            "search_query": resp.search_query, "top_score": resp.top_score,
            "sources": resp.sources, "latency_ms": resp.latency_ms,
            "prompt_tokens": resp.prompt_tokens, "completion_tokens": resp.completion_tokens,
            "checks": check_answer(item, resp),
        }
        if judge is not None and not resp.refused and resp.sources:
            passages = [texts.get(s["chunk_id"], "") for s in resp.sources]
            row["judge"] = judge_faithfulness(judge, resp.answer, passages)
        failed = [k for k, v in row["checks"].items() if (k == "error" and v) or (k != "error" and not v)]
        status = "OK " if not failed else "ÉCHEC"
        lat = resp.latency_ms
        timing = f"{lat.get('total', 0) / 1000:5.1f} s"
        if lat.get("retry_wait"):
            timing += f" (dont {lat['retry_wait'] / 1000:.0f} s d'attente débit)"
        print(f"  [{n:2d}/{len(items)}] {status} {item['qid']:7s} {item['category']:13s} {timing}"
              f"{'  -> ' + ', '.join(failed) if failed else ''}")
        rows.append(row)
        if sleep_seconds and n < len(items):
            time.sleep(sleep_seconds)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-judge", action="store_true", help="Sans LLM juge (plus rapide)")
    parser.add_argument("--limit", type=int, default=None, help="N premières questions seulement")
    args = parser.parse_args()

    from src.rag.ask import build_pipeline  # import ici : nécessite Groq et l'index
    from src.rag.generator import GroqGenerator

    params = load_params()
    d, g, ge = params["data"], params["generation"], params["generation_eval"]
    items = read_jsonl(resolve(d["generation_questions"]))[: args.limit]

    pipeline = build_pipeline()
    judge = None
    if ge.get("use_judge", True) and not args.no_judge:
        judge = GroqGenerator(temperature=0.0, max_tokens=600, reasoning_effort=g.get("reasoning_effort"))

    print(f"Évaluation de la génération : {len(items)} questions, juge={'oui' if judge else 'non'}")
    rows = evaluate(pipeline, items, judge, ge.get("sleep_seconds", 0.0))
    metrics = aggregate(rows, g["price_per_million_input"], g["price_per_million_output"])

    report = {
        "signature": generation_signature(params),
        "llm_model": getattr(pipeline.generator, "model", "unknown"),
        "judge_used": judge is not None,
        "complete_run": args.limit is None,
        "metrics": metrics,
    }
    out = resolve(d["generation_report"])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    write_jsonl(rows, resolve(d["generation_details"]))

    _log_mlflow(report, rows, resolve(d["generation_details"]))

    print("\nRésultats :")
    for k, v in metrics.items():
        if k != "by_category":
            print(f"  {k:30s} {v}")
    print("\nPar catégorie :")
    for cat, m in metrics["by_category"].items():
        print(f"  {cat:15s} n={m['n']:2d}  comportement={m['behavior']}  faits={m['facts']}")
    print(f"\nDétail par question : {resolve(d['generation_details'])}")


def _log_mlflow(report: dict, rows: list[dict], details_path) -> None:
    from src.tracking import setup_mlflow

    mlflow = setup_mlflow("rag-generation")
    if mlflow is None:
        return
    try:
        run_name = f"prompt-{report['signature']['prompt_version']}-{report['llm_model'].split('/')[-1]}"
        with mlflow.start_run(run_name=run_name):
            mlflow.log_params({**report["signature"], "llm_model": report["llm_model"],
                               "judge_used": report["judge_used"], "n_questions": len(rows)})
            mlflow.log_metrics({k: v for k, v in report["metrics"].items()
                                if isinstance(v, (int, float)) and v is not None})
            mlflow.log_artifact(str(details_path))
        print("Run enregistré dans MLflow (expérience rag-generation).")
    except Exception as exc:
        print(f"Avertissement : échec de l'enregistrement MLflow ({type(exc).__name__}) ; "
              "les résultats restent dans reports/.")


if __name__ == "__main__":
    main()
