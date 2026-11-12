"""Quality gates : bloquent la promotion d'une version manifestement dégradée.

Trois familles de contrôles :
  1. Fraîcheur  — le rapport d'évaluation correspond-il aux paramètres actuels,
                  et a-t-il été produit avec le vrai modèle d'embeddings ?
  2. Retrieval  — métriques de la configuration active vs seuils (quality_gates.yaml).
  3. Données    — schéma, doublons, provenance, placeholders, cohérence du jeu
                  d'évaluation (seulement si les fichiers DVC sont présents).

Usage :
  python -m src.quality.gates                 (complet, étape DVC, en local)
  python -m src.quality.gates --report-only   (CI : sans les données DVC)

Code de sortie 1 si un contrôle échoue, ce qui fait échouer dvc repro et la CI.
"""

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

from src.config import active_signature, load_params, resolve
from src.data.cleaning import has_placeholder
from src.data.io import read_jsonl

GATES_PATH = resolve("quality_gates.yaml")
SUMMARY_PATH = resolve("reports/quality_gates.json")


@dataclass
class GateResult:
    family: str
    name: str
    passed: bool
    value: object = None
    threshold: str = ""
    detail: str = ""


def load_gates(path: Path = GATES_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


# 1. Fraîcheur ---------------------------------------------------------------

def check_freshness(report: dict, params: dict) -> list[GateResult]:
    active = report.get("active")
    if not active:
        return [GateResult("freshness", "active_config_present", False,
                           detail="Le rapport n'a pas de section 'active'. Lancez : dvc repro")]
    expected = active_signature(params)
    diffs = {k: (active["config"].get(k), v) for k, v in expected.items()
             if active["config"].get(k) != v}
    results = [GateResult(
        "freshness", "report_matches_params", not diffs,
        detail="" if not diffs else
        "Rapport obsolète (rapport -> params) : "
        + ", ".join(f"{k}: {a} -> {b}" for k, (a, b) in diffs.items())
        + ". Lancez : dvc repro",
    )]
    used = active.get("embedder_used")
    results.append(GateResult(
        "freshness", "real_embedder_used", used == expected["embedding_model"], used,
        f"== {expected['embedding_model']}",
        "" if used == expected["embedding_model"] else
        "Rapport produit avec un embedder factice ou un autre modèle.",
    ))
    return results


# 2. Retrieval ---------------------------------------------------------------

def check_retrieval(metrics: dict, gates: dict) -> list[GateResult]:
    results = []
    for name, rule in gates.items():
        value = metrics.get(name)
        if value is None:
            results.append(GateResult("retrieval", name, False, None, "", "Métrique absente du rapport"))
            continue
        if "min" in rule:
            passed, threshold = value >= rule["min"], f">= {rule['min']}"
        else:
            passed, threshold = value <= rule["max"], f"<= {rule['max']}"
        results.append(GateResult("retrieval", name, passed, value, threshold, rule.get("why", "").strip()))
    return results


# 3. Données ----------------------------------------------------------------

def _unique(items: list, label: str, name: str) -> GateResult:
    dupes = len(items) - len(set(items))
    return GateResult("data", name, dupes == 0, dupes, "== 0", f"{dupes} {label} en double")


def check_data(docs: list[dict], chunks: list[dict], questions: list[dict],
               params: dict, rules: dict) -> list[GateResult]:
    r = []
    n_policy = sum(d.get("doc_type") == "policy" for d in docs)
    r.append(GateResult("data", "min_documents", len(docs) >= rules["min_documents"],
                        len(docs), f">= {rules['min_documents']}"))
    r.append(GateResult("data", "min_policy_sections", n_policy >= rules["min_policy_sections"],
                        n_policy, f">= {rules['min_policy_sections']}"))
    r.append(_unique([d["doc_id"] for d in docs], "doc_id", "unique_doc_ids"))
    r.append(_unique([c["chunk_id"] for c in chunks], "chunk_id", "unique_chunk_ids"))
    r.append(_unique([q["qid"] for q in questions], "qid", "unique_question_ids"))

    missing = [d["doc_id"] for d in docs
               if not d.get("source") or not d.get("doc_type") or not d.get("text", "").strip()]
    r.append(GateResult("data", "documents_schema", not missing, len(missing), "== 0",
                        f"Documents sans source, type ou texte : {missing[:3]}"))

    no_version = [d["doc_id"] for d in docs if d.get("doc_type") == "policy" and not d.get("version")]
    r.append(GateResult("data", "policies_have_version", not no_version, len(no_version), "== 0",
                        "Une politique sans version empêche de résoudre les contradictions"))

    leftovers = [c["chunk_id"] for c in chunks if has_placeholder(c["text"])]
    r.append(GateResult("data", "no_placeholders", not leftovers, len(leftovers), "== 0",
                        f"Placeholders {{{{…}}}} restants : {leftovers[:3]}"))

    doc_ids = {d["doc_id"] for d in docs}
    orphans = [c["chunk_id"] for c in chunks if c.get("doc_id") not in doc_ids]
    r.append(GateResult("data", "chunk_provenance", not orphans, len(orphans), "== 0",
                        f"Chunks sans document d'origine : {orphans[:3]}"))

    limit = params["chunking"]["chunk_size"] + rules.get("max_chunk_words_tolerance", 0)
    too_long = [c["chunk_id"] for c in chunks if len(c["text"].split()) > limit]
    r.append(GateResult("data", "chunk_size_respected", not too_long, len(too_long), "== 0",
                        f"Chunks de plus de {limit} mots : {too_long[:3]}"))

    corpus_intents = {c.get("intent") for c in chunks if c.get("intent")}
    unanswerable = sorted({q["expected_intent"] for q in questions
                           if q["in_scope"] and q["expected_intent"] not in corpus_intents})
    r.append(GateResult("data", "eval_intents_in_corpus", not unanswerable, len(unanswerable), "== 0",
                        f"Intents évalués absents du corpus : {unanswerable[:3]}"))
    return r


# Exécution ------------------------------------------------------------------

def run(report_only: bool = False) -> list[GateResult]:
    params = load_params()
    gates = load_gates()
    report_path = resolve(params["data"]["retrieval_report"])
    if not report_path.exists():
        return [GateResult("freshness", "report_exists", False,
                           detail=f"{report_path.name} introuvable. Lancez : dvc repro")]
    report = json.loads(report_path.read_text(encoding="utf-8"))

    results = check_freshness(report, params)
    results += check_retrieval(report.get("active", {}).get("metrics", {}), gates["retrieval"])

    if not report_only:
        d = params["data"]
        paths = [resolve(d["documents"]), resolve(d["chunks"]), resolve(d["eval_questions"])]
        if all(p.exists() for p in paths):
            docs, chunks, questions = (read_jsonl(p) for p in paths)
            results += check_data(docs, chunks, questions, params, gates["data"])
        else:
            results.append(GateResult("data", "data_available", False,
                                      detail="Données DVC absentes. Lancez : dvc repro"))
    return results


def print_results(results: list[GateResult]) -> None:
    for res in results:
        icon = "PASS" if res.passed else "FAIL"
        value = "" if res.value is None else f" = {res.value}"
        print(f"  [{icon}] {res.family:9s} {res.name:30s}{value} {res.threshold}")
        if not res.passed and res.detail:
            print(f"         -> {res.detail}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-only", action="store_true",
                        help="Contrôle seulement le rapport versionné (CI, sans données DVC)")
    args = parser.parse_args()

    results = run(report_only=args.report_only)
    failed = [r for r in results if not r.passed]
    print(f"\nQuality gates : {len(results) - len(failed)}/{len(results)} réussis\n")
    print_results(results)

    if not args.report_only:
        SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
        SUMMARY_PATH.write_text(json.dumps({
            "passed": not failed,
            "n_checks": len(results),
            "n_failed": len(failed),
            "checks": [asdict(r) for r in results],
        }, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    if failed:
        print(f"\nVERSION REJETÉE : {len(failed)} contrôle(s) en échec.")
        sys.exit(1)
    print("\nVERSION ACCEPTÉE.")


if __name__ == "__main__":
    main()
