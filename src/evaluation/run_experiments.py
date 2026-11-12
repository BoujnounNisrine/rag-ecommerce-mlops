"""Compare les configurations de chunking déclarées dans params.yaml (`experiments`)
et enregistre chaque essai dans MLflow.

Pour chaque configuration : découpage des documents, construction d'un index en
mémoire, évaluation du retrieval sur data/eval/questions.jsonl.

Sorties :
  - un run MLflow par configuration (paramètres, métriques, versions, détail) ;
  - reports/retrieval_metrics.json : métriques de toutes les configurations
    (suivi par DVC comme fichier de métriques).

Usage :
  python -m src.evaluation.run_experiments
  python -m src.evaluation.run_experiments --fake-embedder   (test rapide sans modèle)
"""

import argparse
import json
import tempfile
from pathlib import Path

import yaml

from src.config import active_signature, load_params, resolve
from src.data.chunking import chunk_documents
from src.data.io import read_jsonl, write_jsonl
from src.evaluation.retrieval import evaluate_retrieval
from src.rag.embeddings import HashingEmbedder, load_embedder
from src.rag.index import VectorIndex
from src.tracking import setup_mlflow

EXPERIMENT_NAME = "rag-retrieval"


def data_versions() -> dict:
    """Empreintes DVC des données, pour relier chaque run à une version exacte."""
    lock = resolve("dvc.lock")
    if not lock.exists():
        return {}
    stages = yaml.safe_load(lock.read_text(encoding="utf-8")).get("stages", {})
    versions = {}
    for stage in ("download", "prepare", "make_eval"):
        outs = stages.get(stage, {}).get("outs", [])
        if outs:
            versions[f"data_md5_{stage}"] = outs[0].get("md5", "")[:12]
    return versions


def run_one(cfg: dict, docs: list[dict], questions: list[dict], embedder, params: dict) -> tuple[dict, list]:
    chunks = chunk_documents(docs, cfg["chunk_size"], cfg["chunk_overlap"])
    index = VectorIndex.build(chunks, embedder)
    metrics, details = evaluate_retrieval(
        index, embedder, questions, params["eval"]["ks"], params["retrieval"]["min_score"]
    )
    metrics["n_chunks"] = len(chunks)
    return metrics, details


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fake-embedder", action="store_true",
                        help="Embedder factice (sans PyTorch) pour tester la chaîne")
    args = parser.parse_args()

    params = load_params()
    d = params["data"]
    docs = read_jsonl(resolve(d["documents"]))
    questions = read_jsonl(resolve(d["eval_questions"]))
    embedder = HashingEmbedder() if args.fake_embedder else load_embedder(params)
    versions = data_versions()
    mlflow = setup_mlflow(EXPERIMENT_NAME)

    report = {}
    by_config = {}
    for cfg in params["experiments"]:
        metrics, details = run_one(cfg, docs, questions, embedder, params)
        report[cfg["name"]] = metrics
        by_config[(cfg["chunk_size"], cfg["chunk_overlap"])] = metrics

        if mlflow:
            with mlflow.start_run(run_name=cfg["name"]):
                mlflow.log_params({
                    "chunk_size": cfg["chunk_size"],
                    "chunk_overlap": cfg["chunk_overlap"],
                    "embedding_model": embedder.name,
                    "top_k": params["retrieval"]["top_k"],
                    "min_score": params["retrieval"]["min_score"],
                    "responses_per_intent": params["prepare"]["responses_per_intent"],
                    "seed": params["seed"],
                })
                mlflow.set_tags(versions)
                mlflow.log_metrics({k: v for k, v in metrics.items()
                                    if isinstance(v, (int, float)) and v is not None})
                with tempfile.TemporaryDirectory() as tmp:
                    path = Path(tmp) / "retrieval_details.jsonl"
                    write_jsonl(details, path)
                    mlflow.log_artifact(str(path))

        print(f"\n== {cfg['name']} (size={cfg['chunk_size']}, overlap={cfg['chunk_overlap']}) ==")
        for k, v in metrics.items():
            print(f"  {k:32s} {v}")

    # Configuration active (celle de `chunking`, utilisée par l'assistant) :
    # c'est elle que les quality gates contrôlent.
    active = params["chunking"]
    key = (active["chunk_size"], active["chunk_overlap"])
    if key not in by_config:
        by_config[key], _ = run_one(active, docs, questions, embedder, params)
    report["active"] = {
        "config": active_signature(params),
        "embedder_used": embedder.name,
        "metrics": by_config[key],
    }
    print(f"\nConfiguration active : size={key[0]}, overlap={key[1]}")

    out = resolve(d["retrieval_report"])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nRapport écrit : {out}")
    if mlflow:
        print("Visualisez les runs avec : mlflow ui   puis ouvrez http://127.0.0.1:5000")


if __name__ == "__main__":
    main()
