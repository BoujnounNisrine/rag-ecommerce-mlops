"""Journal structuré des requêtes (monitoring minimal).

Une ligne JSON par requête dans logs/requests.jsonl : latences, erreurs, tokens,
refus, score de retrieval, sources utilisées.

Confidentialité : le texte de la question n'est PAS enregistré. On garde seulement
une empreinte (hash) et la longueur, ce qui permet de compter les questions
répétées sans stocker de données personnelles éventuelles.
"""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def question_fingerprint(question: str) -> str:
    return hashlib.sha256(question.strip().lower().encode("utf-8")).hexdigest()[:16]


class RequestLogger:
    def __init__(self, path: Path | None):
        self.path = Path(path) if path else None

    def log(self, record: dict) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        now = datetime.now(timezone.utc)  # noqa: UP017 (datetime.UTC n'existe pas en Python 3.10)
        record = {"timestamp": now.isoformat(timespec="seconds"), **record}
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
