"""Pipeline RAG : question -> retrieval -> (refus | génération) -> réponse + références.

Le refus hors périmètre se fait AVANT l'appel au LLM : si le meilleur passage a une
similarité inférieure à `min_score`, on renvoie directement le message NO_ANSWER.
Cela évite un appel payant et limite les réponses inventées.
"""

import time
import uuid
from dataclasses import dataclass, field

from src.rag.monitoring import RequestLogger, question_fingerprint
from src.rag.prompt import NO_ANSWER, build_messages


@dataclass
class RAGResponse:
    request_id: str
    answer: str
    refused: bool
    sources: list[dict] = field(default_factory=list)
    top_score: float | None = None
    latency_ms: dict = field(default_factory=dict)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    error: str | None = None
    search_query: str | None = None
    translation_fallback: bool = False


def _finish_latency(latency: dict, t0: float, retry_wait_ms: float) -> None:
    """total = temps réel ; service = total moins l'attente due aux limites de débit."""
    total = (time.perf_counter() - t0) * 1000
    latency["retry_wait"] = round(retry_wait_ms, 1)
    latency["service"] = round(max(total - retry_wait_ms, 0.0), 1)
    latency["total"] = round(total, 1)


def _source_view(result) -> dict:
    c = result.chunk
    return {
        "ref": result.rank,
        "chunk_id": c["chunk_id"],
        "source": c.get("source"),
        "title": c.get("title"),
        "version": c.get("version"),
        "score": round(result.score, 4),
    }


class RAGPipeline:
    def __init__(self, index, embedder, generator, top_k: int = 4, min_score: float = 0.35,
                 logger: RequestLogger | None = None, translator=None):
        if index.meta.get("embedding_model") != embedder.name:
            raise ValueError(
                f"L'index a été construit avec {index.meta.get('embedding_model')} "
                f"mais l'embedder est {embedder.name}. Reconstruisez l'index."
            )
        self.index = index
        self.embedder = embedder
        self.generator = generator
        self.top_k = top_k
        self.min_score = min_score
        self.logger = logger or RequestLogger(None)
        self.translator = translator

    def retrieve(self, question: str, k: int | None = None):
        vector = self.embedder.encode([question])[0]
        return self.index.search(vector, k or self.top_k)

    def ask(self, question: str) -> RAGResponse:
        request_id = uuid.uuid4().hex[:12]
        question = (question or "").strip()
        t0 = time.perf_counter()

        if not question:
            response = RAGResponse(request_id, NO_ANSWER, refused=True, error="empty_question")
            self._log(response, question)
            return response

        # 1. Traduction de la requête de recherche (optionnelle)
        search_query, latency = question, {}
        tr_prompt = tr_completion = 0
        fallback = False
        retry_wait = 0.0  # attente due aux limites de débit (pas du temps de calcul)
        if self.translator is not None:
            tr = self.translator.translate(question)
            search_query, fallback = tr.text, tr.used_fallback
            tr_prompt, tr_completion = tr.prompt_tokens, tr.completion_tokens
            latency["translation"] = tr.latency_ms
            retry_wait += getattr(tr, "retry_wait_ms", 0.0)

        # 2. Retrieval
        t1 = time.perf_counter()
        results = self.retrieve(search_query)
        latency["retrieval"] = round((time.perf_counter() - t1) * 1000, 1)
        top_score = results[0].score if results else None
        sources = [_source_view(r) for r in results]

        # 3. Refus avant le LLM si aucun passage n'est assez proche
        if top_score is None or top_score < self.min_score:
            _finish_latency(latency, t0, retry_wait)
            response = RAGResponse(
                request_id, NO_ANSWER, refused=True, sources=sources, top_score=top_score,
                latency_ms=latency, prompt_tokens=tr_prompt, completion_tokens=tr_completion,
                search_query=search_query, translation_fallback=fallback,
            )
            self._log(response, question)
            return response

        # 4. Génération avec la question ORIGINALE (la réponse suit sa langue)
        t2 = time.perf_counter()
        try:
            generation = self.generator.generate(build_messages(question, results))
            answer, error = generation.text, None
            if not answer:
                answer, error = NO_ANSWER, "empty_answer"
        except Exception as exc:  # erreur réseau, quota, clé invalide…
            generation, answer, error = None, NO_ANSWER, type(exc).__name__
        latency["generation"] = round((time.perf_counter() - t2) * 1000, 1)
        if generation is not None:
            retry_wait += getattr(generation, "retry_wait_ms", 0.0)
        _finish_latency(latency, t0, retry_wait)

        response = RAGResponse(
            request_id=request_id,
            answer=answer,
            refused=answer.strip() == NO_ANSWER,
            sources=sources,
            top_score=top_score,
            latency_ms=latency,
            prompt_tokens=tr_prompt + (generation.prompt_tokens if generation else 0),
            completion_tokens=tr_completion + (generation.completion_tokens if generation else 0),
            error=error,
            search_query=search_query,
            translation_fallback=fallback,
        )
        self._log(response, question)
        return response

    def _log(self, response: RAGResponse, question: str) -> None:
        self.logger.log(
            {
                "request_id": response.request_id,
                "question_hash": question_fingerprint(question),
                "question_length": len(question),
                "refused": response.refused,
                "top_score": None if response.top_score is None else round(response.top_score, 4),
                "sources": [s["chunk_id"] for s in response.sources],
                "latency_ms": response.latency_ms,
                "prompt_tokens": response.prompt_tokens,
                "completion_tokens": response.completion_tokens,
                "embedding_model": self.embedder.name,
                "generator": getattr(self.generator, "name", "unknown"),
                "translated": self.translator is not None,
                "translation_fallback": response.translation_fallback,
                "error": response.error,
            }
        )
