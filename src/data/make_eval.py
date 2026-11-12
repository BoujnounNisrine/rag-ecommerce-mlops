"""Étape DVC `make_eval` : construit le jeu de questions d'évaluation.

- Questions « dans le périmètre » : tirées de la colonne `instruction` de Bitext,
  stratifiées par intent. L'intent sert de vérité terrain pour Recall@k :
  un retrieval est réussi si un chunk retrouvé porte le même intent.
- Questions « hors périmètre » : lues dans data/eval/out_of_scope.txt.
  L'assistant doit refuser d'y répondre.

Usage : python -m src.data.make_eval
"""

import pandas as pd

from src.config import load_params, resolve
from src.data.cleaning import clean_text
from src.data.io import write_jsonl
from src.data.prepare import load_bitext


def in_scope_questions(df: pd.DataFrame, per_intent: int, seed: int) -> list[dict]:
    df = df.dropna(subset=["instruction", "intent"]).copy()
    df["question"] = df["instruction"].map(clean_text)
    df = df[df["question"].str.len() > 0].drop_duplicates(subset="question")

    out = []
    for intent, group in sorted(df.groupby("intent"), key=lambda kv: kv[0]):
        sample = group.sample(n=min(per_intent, len(group)), random_state=seed)
        for i, (_, row) in enumerate(sample.iterrows()):
            out.append(
                {
                    "qid": f"in-{intent}-{i}",
                    "question": row["question"],
                    "expected_intent": intent,
                    "category": row["category"],
                    "in_scope": True,
                    "flags": row.get("flags"),
                }
            )
    return out


def out_of_scope_questions(path) -> list[dict]:
    lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines()]
    questions = [ln for ln in lines if ln and not ln.startswith("#")]
    return [
        {"qid": f"oos-{i}", "question": q, "expected_intent": None, "category": None, "in_scope": False}
        for i, q in enumerate(questions)
    ]


def main() -> None:
    params = load_params()
    d = params["data"]
    df = load_bitext(resolve(d["raw_bitext"]))

    questions = in_scope_questions(df, params["eval"]["questions_per_intent"], params["seed"])
    questions += out_of_scope_questions(resolve(d["out_of_scope"]))

    n = write_jsonl(questions, resolve(d["eval_questions"]))
    n_oos = sum(not q["in_scope"] for q in questions)
    print(f"{n} questions écrites ({n - n_oos} dans le périmètre, {n_oos} hors périmètre)")


if __name__ == "__main__":
    main()
