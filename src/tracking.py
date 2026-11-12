"""Configuration commune du suivi MLflow.

- Base de suivi : `sqlite:///mlflow.db` à la racine du projet (le stockage en
  dossier `./mlruns` est refusé par les versions récentes de MLflow).
- MLFLOW_TRACKING_URI (dans .env) peut pointer vers une autre base ou un serveur ;
  une ancienne valeur `file:` est ignorée avec un avertissement.
- Le suivi ne doit jamais faire perdre une évaluation : en cas de problème, on
  affiche un avertissement et l'évaluation continue sans MLflow.
"""

import os

from src.config import ROOT

DEFAULT_URI = f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}"


def tracking_uri() -> str:
    uri = os.getenv("MLFLOW_TRACKING_URI", "").strip()
    if not uri:
        return DEFAULT_URI
    if uri.startswith("file:") or "://" not in uri:
        print(f"Avertissement : MLFLOW_TRACKING_URI={uri} (stockage en dossier) n'est plus "
              f"supporté par MLflow, utilisation de {DEFAULT_URI}. Corrigez votre .env.")
        return DEFAULT_URI
    return uri


def setup_mlflow(experiment: str):
    """Renvoie le module mlflow prêt à l'emploi, ou None si indisponible."""
    try:
        import mlflow
    except ImportError:
        print("MLflow n'est pas installé : résultats écrits seulement dans reports/.")
        return None
    try:
        mlflow.set_tracking_uri(tracking_uri())
        mlflow.set_experiment(experiment)
        return mlflow
    except Exception as exc:  # base verrouillée, version incompatible…
        print(f"Avertissement : suivi MLflow désactivé ({type(exc).__name__}: {exc})")
        return None
