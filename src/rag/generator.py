"""Générateurs de réponses.

- GroqGenerator : appelle l'API Groq (import paresseux, clé lue dans .env).
- EchoGenerator : générateur factice pour les tests, sans réseau.

Un générateur renvoie un GenerationResult avec le texte et les tokens consommés,
utilisés par le monitoring pour estimer le coût.
"""

import os
import time
from dataclasses import dataclass

RETRYABLE_ERRORS = {"RateLimitError", "APITimeoutError", "APIConnectionError", "InternalServerError"}


def call_with_retry(fn, max_attempts: int = 4, base_delay: float = 2.0, sleep=time.sleep,
                    stats: dict | None = None):
    """Réessaie les erreurs transitoires (limite de débit, réseau) avec attente exponentielle.

    Les autres erreurs (clé invalide, modèle introuvable…) sont relancées immédiatement :
    réessayer ne servirait à rien.

    Si `stats` est fourni, on y enregistre le nombre de nouvelles tentatives et le temps
    d'attente, pour distinguer la latence du service de l'attente due aux limites de débit.
    """
    if stats is not None:
        stats.setdefault("retries", 0)
        stats.setdefault("retry_wait_ms", 0.0)
    for attempt in range(1, max_attempts + 1):
        try:
            return fn()
        except Exception as exc:
            if type(exc).__name__ not in RETRYABLE_ERRORS or attempt == max_attempts:
                raise
            delay = base_delay * 2 ** (attempt - 1)
            if stats is not None:
                stats["retries"] += 1
                stats["retry_wait_ms"] += delay * 1000
            sleep(delay)


@dataclass
class GenerationResult:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    model: str = ""
    retries: int = 0
    retry_wait_ms: float = 0.0


class EchoGenerator:
    """Renvoie une réponse fixe citant la source [1] (tests uniquement)."""

    name = "echo"

    def __init__(self, text: str = "Here is the answer [1]."):
        self.text = text
        self.calls = 0

    def generate(self, messages: list[dict]) -> GenerationResult:
        self.calls += 1
        return GenerationResult(text=self.text, model=self.name)


class GroqGenerator:
    def __init__(self, temperature: float = 0.0, max_tokens: int = 400, model: str | None = None,
                 reasoning_effort: str | None = None):
        from dotenv import load_dotenv
        from groq import Groq  # import paresseux

        load_dotenv()
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY manquante. Copiez .env.example en .env et ajoutez votre clé."
            )
        self.model = model or os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
        self.name = f"groq/{self.model}"
        self.temperature = temperature
        self.max_tokens = max_tokens
        # Uniquement pour les modèles qui raisonnent (gpt-oss…) : "low" = plus rapide
        # et moins de tokens. Mettre null dans params.yaml pour les autres modèles.
        self.reasoning_effort = reasoning_effort
        # max_retries=0 : la bibliothèque groq réessaie sinon en silence, ce qui rend
        # les attentes invisibles. Toutes les tentatives passent par call_with_retry,
        # qui les mesure.
        self._client = Groq(api_key=api_key, max_retries=0)

    def generate(self, messages: list[dict]) -> GenerationResult:
        kwargs = {}
        if self.reasoning_effort:
            kwargs["reasoning_effort"] = self.reasoning_effort
        stats = {}
        response = call_with_retry(lambda: self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            **kwargs,
        ), stats=stats)
        usage = response.usage
        content = response.choices[0].message.content or ""  # peut être vide si tokens épuisés
        return GenerationResult(
            text=content.strip(),
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            model=self.model,
            retries=stats["retries"],
            retry_wait_ms=stats["retry_wait_ms"],
        )
