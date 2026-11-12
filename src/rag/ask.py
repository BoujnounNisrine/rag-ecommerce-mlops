"""Interface en ligne de commande pour interroger l'assistant.

Usage :
  python -m src.rag.ask "Comment puis-je retourner un article ?"
  python -m src.rag.ask            (mode interactif, tapez `exit` pour quitter)
"""

import sys

from src.config import load_params, resolve
from src.rag.embeddings import load_embedder
from src.rag.generator import GroqGenerator
from src.rag.index import VectorIndex
from src.rag.monitoring import RequestLogger
from src.rag.pipeline import RAGPipeline
from src.rag.translation import QueryTranslator


def build_pipeline() -> RAGPipeline:
    params = load_params()
    d, r, g = params["data"], params["retrieval"], params["generation"]
    effort = g.get("reasoning_effort")

    translator = None
    if r.get("translate_query", False):
        translator = QueryTranslator(
            GroqGenerator(temperature=0.0, max_tokens=g.get("translation_max_tokens", 300),
                          reasoning_effort=effort)
        )

    return RAGPipeline(
        index=VectorIndex.load(resolve(d["index_dir"])),
        embedder=load_embedder(params),
        generator=GroqGenerator(temperature=g["temperature"], max_tokens=g["max_tokens"],
                                reasoning_effort=effort),
        top_k=r["top_k"],
        min_score=r["min_score"],
        logger=RequestLogger(resolve(d["request_log"])),
        translator=translator,
    )


def print_response(resp, question: str) -> None:
    print(f"\n{resp.answer}\n")
    if resp.search_query and resp.search_query != question:
        print(f"Requête de recherche (traduite) : {resp.search_query}")
    if resp.sources:
        print("Sources :" if not resp.refused else "Passages consultés (jugés insuffisants) :")
        for s in resp.sources:
            version = f", version {s['version']}" if s["version"] else ""
            print(f"  [{s['ref']}] {s['source']} — {s['title']}{version} (score {s['score']:.2f})")
    score = "n/a" if resp.top_score is None else f"{resp.top_score:.2f}"
    details = ", ".join(f"{k} {v:.0f} ms" for k, v in resp.latency_ms.items() if k != "total")
    print(f"\n(latence {resp.latency_ms.get('total', 0):.0f} ms [{details}], meilleur score {score}"
          f"{', refus' if resp.refused else ''}{', erreur ' + resp.error if resp.error else ''})")


def main() -> None:
    pipeline = build_pipeline()
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
        print_response(pipeline.ask(question), question)
        return
    print("Assistant ShopFès — posez votre question (exit pour quitter)")
    while True:
        question = input("\n> ").strip()
        if question.lower() in {"exit", "quit"}:
            break
        print_response(pipeline.ask(question), question)


if __name__ == "__main__":
    main()
