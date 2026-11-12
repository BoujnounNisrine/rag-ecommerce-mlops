# Assistant RAG Support E-commerce — Mini-projet MLOps

Mini-projet MLOps (ENSAF / ISDIA3, 2026–2027) — **Projet 1 : De l'étude de cas à l'industrialisation d'un assistant RAG.**

Assistant de support client pour **ShopFès**, une boutique en ligne fictive. Il répond aux questions
sur les commandes, livraisons, retours, remboursements et comptes, à partir d'une documentation de
référence, cite ses sources et refuse les questions hors périmètre.

## Données

| Source | Contenu | Licence | Rôle |
|---|---|---|---|
| [Bitext Customer Support](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset) | ~27 000 paires question/réponse, 27 intents | CDLA-Sharing 1.0 | Corpus (réponses) + questions d'évaluation |
| `data/raw/policies/*.md` | Politiques de retour, livraison, garantie de ShopFès | Rédigées pour le projet | Corpus en Markdown, versionné |

`warranty_old.md` est **volontairement obsolète** (retour sous 14 jours contre 30 jours dans
`return_policy.md`) pour tester le comportement face à des documents contradictoires.

## Architecture

```
Bitext CSV + politiques .md
        │  download → prepare (nettoyage, provenance)
        ▼
documents.jsonl ──chunk──► chunks.jsonl ──► embeddings + index FAISS unique
                                                     │
question ─► retrieval top-k ─► contexte ─► LLM (Groq) ─► réponse + références
```

RAG monolithique, **un seul index vectoriel**, comme demandé par le cahier des charges.

## Structure

```
├── params.yaml            # toute la configuration (chunking, seed, chemins…)
├── dvc.yaml               # pipeline : download → prepare → chunk → make_eval
├── src/
│   ├── config.py          # chargement + validation de params.yaml
│   └── data/
│       ├── download.py    # étape download
│       ├── cleaning.py    # remplacement des placeholders {{…}} de Bitext
│       ├── prepare.py     # étape prepare : corpus + métadonnées de provenance
│       ├── chunking.py    # découpage en fenêtre glissante
│       ├── chunk.py       # étape chunk
│       └── make_eval.py   # étape make_eval : questions annotées
├── data/
│   ├── raw/policies/      # documents de politique (Git)
│   └── eval/out_of_scope.txt
├── tests/                 # pytest, sans réseau
└── .github/workflows/ci.yml
```

## Installation

```bash
git clone <votre-depot>
cd rag-ecommerce-mlops
python -m venv .venv
source .venv/bin/activate        # Windows : .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env             # puis ajoutez votre clé GROQ_API_KEY
```

## Exécution du pipeline de données

```bash
dvc init                 # une seule fois
dvc repro                # exécute download → prepare → chunk → make_eval
```

Ou étape par étape :

```bash
python -m src.data.download
python -m src.data.prepare
python -m src.data.chunk
python -m src.data.make_eval
```

Sorties :
- `data/processed/documents.jsonl` — un document par ligne, avec `doc_id`, `source`, `intent`, `version`…
- `data/processed/chunks.jsonl` — chunks avec la provenance de leur document
- `data/eval/questions.jsonl` — questions dans le périmètre (avec `expected_intent`) et hors périmètre

## Tests et qualité

```bash
ruff check .
pytest -q
```

La CI GitHub Actions exécute à chaque push : lint, validation de `params.yaml`, tests unitaires,
et vérification qu'aucun secret n'est versionné.

## Évaluation (prévue)

- **Recall@k** : un retrieval est réussi si l'un des k chunks retrouvés porte le même `intent`
  que la question.
- **Hors périmètre** : taux de refus correct sur `out_of_scope.txt`.
- **Expériences** : `small_chunks` (120/20) vs `large_chunks` (300/60), suivies dans MLflow.

## Avancement

- [x] Structure du dépôt, configuration, pipeline DVC
- [x] Ingestion, nettoyage, chunking, jeu d'évaluation
- [x] Tests unitaires et CI
- [ ] Embeddings + index FAISS
- [ ] Pipeline RAG (retrieval + génération Groq + références)
- [ ] Évaluation + MLflow + quality gates
- [ ] Monitoring (latence, erreurs, coût)
- [ ] Rapport de risques et gouvernance
