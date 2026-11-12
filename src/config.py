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


def load_params(path: Path = PARAMS_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        params = yaml.safe_load(f)
    validate(params)
    return params


def resolve(relative: str) -> Path:
    """Chemin absolu à partir de la racine du dépôt."""
    return ROOT / relative
