"""Étape DVC `prepare` : construit le corpus documentaire.

Deux sources :
  1. Bitext (CSV) — on garde quelques réponses distinctes par intent ;
  2. les politiques de ShopFès (Markdown), découpées par section `##`.

Sortie : data/processed/documents.jsonl, un document par ligne avec sa provenance.
Usage : python -m src.data.prepare
"""

import re
from pathlib import Path

import pandas as pd

from src.config import load_params, resolve
from src.data.cleaning import clean_text
from src.data.io import write_jsonl

REQUIRED_COLUMNS = {"instruction", "intent", "category", "response"}


def load_bitext(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes dans {path.name} : {missing}")
    return df


def bitext_documents(
    df: pd.DataFrame, responses_per_intent: int, min_words: int, seed: int
) -> list[dict]:
    df = df.dropna(subset=["response", "intent"]).copy()
    df["clean"] = df["response"].map(clean_text)
    df = df[df["clean"].str.split().str.len() >= min_words]
    df = df.drop_duplicates(subset="clean")

    docs = []
    for intent, group in sorted(df.groupby("intent"), key=lambda kv: kv[0]):
        sample = group.sample(n=min(responses_per_intent, len(group)), random_state=seed)
        for i, (_, row) in enumerate(sample.iterrows()):
            docs.append(
                {
                    "doc_id": f"bitext-{intent}-{i}",
                    "source": "bitext",
                    "doc_type": "faq_response",
                    "title": intent.replace("_", " "),
                    "intent": intent,
                    "category": row["category"],
                    "text": row["clean"],
                }
            )
    return docs


def split_markdown_sections(text: str) -> list[tuple[str, str]]:
    """Découpe un Markdown en (titre de section, contenu) sur les titres `##`."""
    parts = re.split(r"^##\s+(.+)$", text, flags=re.MULTILINE)
    sections = []
    for title, body in zip(parts[1::2], parts[2::2], strict=True):
        body = body.strip()
        if body:
            sections.append((title.strip(), body))
    return sections


def policy_documents(policies_dir: Path) -> list[dict]:
    docs = []
    for md in sorted(policies_dir.glob("*.md")):
        text = md.read_text(encoding="utf-8")
        version = re.search(r"^Version\s*:\s*(.+)$", text, flags=re.MULTILINE)
        for i, (title, body) in enumerate(split_markdown_sections(text)):
            docs.append(
                {
                    "doc_id": f"policy-{md.stem}-{i}",
                    "source": f"policies/{md.name}",
                    "doc_type": "policy",
                    "title": title,
                    "intent": None,
                    "category": "POLICY",
                    "version": version.group(1).strip() if version else None,
                    "text": clean_text(body),
                }
            )
    return docs


def main() -> None:
    params = load_params()
    p, d = params["prepare"], params["data"]

    df = load_bitext(resolve(d["raw_bitext"]))
    docs = bitext_documents(df, p["responses_per_intent"], p["min_response_words"], params["seed"])
    docs += policy_documents(resolve(d["policies_dir"]))

    n = write_jsonl(docs, resolve(d["documents"]))
    n_policy = sum(doc["doc_type"] == "policy" for doc in docs)
    print(f"{n} documents écrits ({n - n_policy} Bitext, {n_policy} sections de politique)")


if __name__ == "__main__":
    main()
