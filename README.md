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
documents.jsonl ──chunk──► chunks.jsonl ──► embeddings + index vectoriel unique
                                                     │
question ─► retrieval top-k ─► contexte ─► LLM (Groq) ─► réponse + références
```

RAG monolithique, **un seul index vectoriel**, comme demandé par le cahier des charges.

## Structure

```
├── params.yaml            # toute la configuration (chunking, seed, chemins…)
├── dvc.yaml               # download → prepare → chunk → make_eval → build_index → evaluate_retrieval
├── src/
│   ├── config.py          # chargement + validation de params.yaml
│   ├── data/
│   │   ├── download.py    # étape download
│   │   ├── cleaning.py    # remplacement des placeholders {{…}} de Bitext
│   │   ├── prepare.py     # étape prepare : corpus + métadonnées de provenance
│   │   ├── chunking.py    # découpage en fenêtre glissante
│   │   ├── chunk.py       # étape chunk
│   │   └── make_eval.py   # étape make_eval : questions annotées
│   ├── rag/
│   │   ├── embeddings.py  # sentence-transformers (+ embedder factice pour les tests)
│   │   ├── index.py       # index vectoriel unique, recherche cosinus exacte
│   │   ├── build_index.py # étape build_index
│   │   ├── prompt.py      # prompt avec citations et garde-fous
│   │   ├── generator.py   # appel à l'API Groq
│   │   ├── pipeline.py    # retrieval → refus ou génération → réponse + sources
│   │   ├── monitoring.py  # journal JSONL des requêtes (sans le texte des questions)
│   │   └── ask.py         # interface en ligne de commande
│   └── evaluation/
│       ├── retrieval.py        # Recall@k, MRR, seuil hors périmètre
│       └── run_experiments.py  # comparaison des configurations + MLflow
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

## Utiliser l'assistant

```bash
dvc repro build_index                                   # construit l'index (télécharge le modèle la 1re fois)
python -m src.rag.ask "How can I return an item?"       # une question
python -m src.rag.ask                                   # mode interactif
```

La réponse, en français, cite ses sources (`[1]`, `[2]`…), avec le fichier, le titre et la version.
Si le meilleur passage a une similarité inférieure à `retrieval.min_score`, l'assistant
refuse **sans appeler le LLM de réponse**.

**Questions en français** : les documents et le modèle d'embeddings sont en anglais. Avec
`retrieval.translate_query: true`, la question est traduite en anglais (appel Groq court)
pour la recherche uniquement ; le LLM reçoit la question originale. En cas d'échec de la
traduction, la recherche se fait avec la question originale. Un modèle d'embeddings
multilingue a été testé et rejeté (moins bon en anglais comme en français).

## Évaluation et expériences

```bash
dvc repro evaluate_retrieval      # ou : python -m src.evaluation.run_experiments
mlflow ui                         # puis http://127.0.0.1:5000
```

- **Recall@k / MRR** : un retrieval est réussi si l'un des k chunks porte le même `intent`
  que la question. Les chunks de politique n'ont pas d'intent, donc Recall@k est une borne basse.
- **Hors périmètre** : taux de refus sur `out_of_scope.txt` et taux de faux refus sur les
  questions du périmètre.
- **Seuil `min_score`** : `suggested_min_score` est le seuil qui maximise l'exactitude
  équilibrée (accepter le périmètre, refuser le hors périmètre). Il justifie la valeur
  retenue dans `params.yaml`.
- **Expériences** : `small_chunks` (120/20) vs `large_chunks` (300/60). Chaque run MLflow
  enregistre paramètres, métriques, empreintes DVC des données et détail par question.

## Quality gates

Une version est **rejetée** si un seul critère de `quality_gates.yaml` échoue. Chaque seuil
y est justifié à partir de la baseline mesurée.

| Famille | Contrôles | Où |
|---|---|---|
| Code | lint (ruff), tests unitaires, validation de `params.yaml` | CI |
| Sécurité | aucun `.env` ni clé API versionnés | CI |
| Fraîcheur | le rapport d'évaluation correspond à `params.yaml` et au vrai modèle | CI + DVC |
| Retrieval | Recall@1, Recall@3, MRR, faux refus, refus hors périmètre, taille du jeu d'évaluation | CI + DVC |
| Données | doublons, schéma, versions des politiques, placeholders, provenance des chunks, taille des chunks, intents évalués présents dans le corpus | DVC |

```bash
python -m src.quality.gates                 # tous les contrôles (dernière étape de dvc repro)
python -m src.quality.gates --report-only   # contrôles exécutés par la CI
```

Conséquence : après toute modification de `params.yaml`, il faut relancer `dvc repro` et
committer le nouveau `reports/retrieval_metrics.json`, sinon la CI échoue.

## Monitoring

Chaque requête ajoute une ligne à `logs/requests.jsonl` : latence (retrieval, génération,
total), tokens consommés, refus, erreur, meilleur score et sources. Le texte de la question
n'est pas stocké, seulement une empreinte et sa longueur.

## Avancement

- [x] Structure du dépôt, configuration, pipeline DVC
- [x] Ingestion, nettoyage, chunking, jeu d'évaluation
- [x] Tests unitaires et CI
- [x] Embeddings + index vectoriel
- [x] Pipeline RAG (retrieval + génération Groq + références + refus)
- [x] Évaluation du retrieval + MLflow
- [x] Journal de monitoring
- [ ] Évaluation de la génération (fidélité, exactitude)
- [x] Quality gates (CI + DVC)
- [ ] Tableau de bord de monitoring
- [ ] Rapport de risques et gouvernance
