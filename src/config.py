"""Chargement et validation de params.yaml."""

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
PARAMS_PATH = ROOT / "params.yaml"


class ConfigError(ValueError):
    """Configuration invalide."""


def validate(params: dict) -> None:
    """Lève ConfigError si la configuration est incohérente."""
    for key in ("seed", "data", "prepare", "chunking", "eval"):
        if key not in params:
            raise ConfigError(f"Clé manquante dans params.yaml : {key}")

    configs = [params["chunking"], *params.get("experiments", [])]
    for cfg in configs:
        size, overlap = cfg["chunk_size"], cfg["chunk_overlap"]
        if size <= 0:
            raise ConfigError(f"chunk_size doit être > 0 (reçu {size})")
        if not 0 <= overlap < size:
            raise ConfigError(
                f"chunk_overlap doit être dans [0, chunk_size[ (reçu {overlap}, size={size})"
            )

    if params["prepare"]["responses_per_intent"] < 1:
        raise ConfigError("responses_per_intent doit être >= 1")
    if params["eval"]["questions_per_intent"] < 1:
        raise ConfigError("questions_per_intent doit être >= 1")

    retrieval = params.get("retrieval")
    if retrieval is not None:
        if retrieval["top_k"] < 1:
            raise ConfigError("retrieval.top_k doit être >= 1")
        if not -1.0 <= retrieval["min_score"] <= 1.0:
            raise ConfigError("retrieval.min_score doit être dans [-1, 1] (similarité cosinus)")

    generation = params.get("generation")
    if generation is not None:
        if not 0.0 <= generation["temperature"] <= 2.0:
            raise ConfigError("generation.temperature doit être dans [0, 2]")
        if generation["max_tokens"] < 1:
            raise ConfigError("generation.max_tokens doit être >= 1")
        if generation.get("reasoning_effort") not in (None, "low", "medium", "high"):
            raise ConfigError("generation.reasoning_effort doit être null, low, medium ou high")


def load_params(path: Path = PARAMS_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        params = yaml.safe_load(f)
    validate(params)
    return params


def resolve(relative: str) -> Path:
    """Chemin absolu à partir de la racine du dépôt."""
    return ROOT / relative
