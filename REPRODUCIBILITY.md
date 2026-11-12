# Dossier de reproductibilité

Ce dossier permet de reconstruire, à partir d'un dépôt vide, les données, l'index,
les évaluations et les décisions des quality gates du projet, et de vérifier que
le résultat est identique à celui obtenu le **2026-11-12**.

**Ce qui se reproduit à l'identique** : corpus, chunks, jeu d'évaluation, index
vectoriel et toutes les métriques de retrieval (calcul déterministe sur CPU).
**Ce qui se reproduit à une tolérance près** : les métriques de génération, car elles
dépendent d'une API externe (Groq) dont le modèle n'est pas parfaitement déterministe,
même à température 0. Les quality gates définissent cette tolérance.

---

## 1. Environnement

| Élément | Version utilisée | Remarque |
|---|---|---|
| Système | Windows, PowerShell | Développement et résultats de référence |
| Python | 3.13.2 | La CI utilise Python 3.11 (Ubuntu) : les tests passent sur les deux |
| PyTorch | 2.15.0 (CPU) | Installé depuis l'index CPU de PyTorch, sans GPU |
| Bibliothèques | voir `requirements-lock.txt` | Versions figées des dépendances directes |
| Modèle d'embeddings | `sentence-transformers/all-MiniLM-L6-v2` | Téléchargé depuis Hugging Face au premier lancement (~90 Mo) |
| Modèle de génération | `openai/gpt-oss-20b` (Groq) | Nom lu dans `.env` (`GROQ_MODEL`) |
| Suivi des expériences | MLflow 3.18.0, base `sqlite:///mlflow.db` | Base locale, non versionnée |

Paramètres de génération : température 0, `reasoning_effort: low`,
`max_tokens: 1000`, prompt **v2** (`src/rag/prompt.py`, constante `PROMPT_VERSION`).

## 2. Reconstruction pas à pas

```powershell
# 1. Récupérer le code
git clone https://github.com/BoujnounNisrine/rag-ecommerce-mlops.git
cd rag-ecommerce-mlops

# 2. Environnement virtuel
python -m venv .venv
.venv\Scripts\activate            # Linux / macOS : source .venv/bin/activate

# 3. Dépendances aux versions exactes (PyTorch CPU d'abord)
pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-lock.txt

# 4. Secret : clé API Groq (gratuite sur console.groq.com)
copy .env.example .env            # Linux / macOS : cp .env.example .env
#    puis renseigner GROQ_API_KEY dans .env

# 5. Reconstruire toute la chaîne
dvc repro

# 6. Vérifier
pytest -q
python -m src.quality.gates
```

`dvc repro` exécute, dans l'ordre et seulement si nécessaire :

| Étape | Rôle | Durée indicative |
|---|---|---|
| `download` | Télécharge le CSV Bitext (19,2 Mo) | quelques secondes |
| `prepare` | Nettoie les placeholders, construit le corpus avec sa provenance | < 5 s |
| `chunk` | Découpe en chunks de 300 mots (chevauchement 60) | < 1 s |
| `make_eval` | Construit les 138 questions d'évaluation du retrieval | < 5 s |
| `build_index` | Calcule les embeddings et l'index vectoriel | ~5 s (+ téléchargement du modèle) |
| `evaluate_retrieval` | Compare les configurations, écrit le rapport, enregistre dans MLflow | ~15 s |
| `evaluate_generation` | Pose les 28 questions annotées à l'assistant (~85 appels Groq) | 1 à 3 min |
| `quality_gates` | Applique les 31 contrôles ; échoue si un seul échoue | < 2 s |

## 3. Empreintes des données (extraites de `dvc.lock`)

Une reconstruction réussie produit exactement ces empreintes MD5. Après `dvc repro`,
la commande `dvc status` doit répondre `Data and pipelines are up to date.`

| Fichier | MD5 | Taille |
|---|---|---|
| `data/raw/bitext.csv` (source) | `ffbbbde8a9481dfc8943f264b8b9c9e6` | 19 202 474 octets |
| `data/raw/policies/` (3 fichiers) | `b4a7ae311daabac121cb10e4a41cced7.dir` | 2 063 octets |
| `data/processed/documents.jsonl` | `a9a66e1a1c10f6bde96da07f587b9315` | 114 555 octets |
| `data/processed/chunks.jsonl` | `d2ae51a00a63e18a98f7d3a8081ac8de` | 125 678 octets |
| `data/eval/questions.jsonl` | `8745e3c121eadb3317d44edd5058bcd2` | 25 050 octets |
| `data/eval/out_of_scope.txt` | `cc5cc87283755bb4c4ae87bb099fbe74` | 1 719 octets |
| `data/eval/generation_questions.jsonl` | `7de85ac84d357f443eef08223ea8b891` | 6 647 octets |
| `artifacts/index/` (3 fichiers) | `3cc0164857b806d866c58ab2b7de8ce9.dir` | 350 209 octets |

Source du CSV : [Bitext Customer Support Dataset](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset),
fichier `Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv`, licence CDLA-Sharing 1.0.

## 4. Paramètres de référence

Tous les paramètres sont dans `params.yaml`, versionné avec Git. Les valeurs qui
déterminent les résultats :

| Paramètre | Valeur | Justification |
|---|---|---|
| `seed` | 42 | Échantillonnage du corpus et des questions |
| `prepare.responses_per_intent` | 5 | 27 intents × 5 = 135 réponses Bitext |
| `chunking` | 300 mots, chevauchement 60 | Meilleure configuration mesurée (`large_chunks`) |
| `embeddings.model` | `all-MiniLM-L6-v2` | Meilleur que le modèle multilingue testé |
| `retrieval.top_k` | 4 | Passages fournis au LLM |
| `retrieval.min_score` | 0,36 | Seuil suggéré 0,3645 arrondi vers le bas |
| `retrieval.translate_query` | `true` | Questions en français sur des documents en anglais |
| `generation.temperature` | 0 | Réponses aussi stables que possible |

## 5. Résultats attendus

### Volumes (identiques)

| Élément | Valeur |
|---|---|
| Documents | 145 (135 Bitext + 10 sections de politique) |
| Chunks | 146 |
| Questions d'évaluation du retrieval | 138 (108 dans le périmètre + 30 hors périmètre) |
| Questions d'évaluation de la génération | 28 |
| Tests automatisés | tous réussis (`pytest -q`, aucun échec) |

### Retrieval (identiques, configuration active `large_chunks`)

| Métrique | Valeur attendue |
|---|---|
| Recall@1 | 0,8241 |
| Recall@3 | 0,9444 |
| Recall@5 | 0,9722 |
| MRR | 0,8861 |
| Refus à tort (périmètre) | 0,0 |
| Refus hors périmètre | 0,9333 |
| Seuil suggéré | 0,3645 |

### Génération (à une tolérance près)

| Métrique | Valeur de référence | Seuil de la gate |
|---|---|---|
| Comportement correct | 0,9643 | ≥ 0,89 |
| Exactitude des faits | 0,95 | ≥ 0,85 |
| Citations présentes | 1,0 | ≥ 0,95 |
| Citations valides | 1,0 | ≥ 0,95 |
| Fidélité (LLM juge) | 0,9474 | ≥ 0,85 |
| Contenu interdit | 0,0 | = 0 |
| Latence de service P95 | 3 656 ms | ≤ 8 000 ms |

**Critère de réussite de la reconstruction** : `python -m src.quality.gates` affiche
**31/31 réussis** et **VERSION ACCEPTÉE**.

## 6. Écarts possibles et leur cause

| Écart observé | Cause | Que faire |
|---|---|---|
| Métriques de génération légèrement différentes | Le modèle Groq n'est pas parfaitement déterministe, même à température 0 | Normal tant que les gates passent |
| Latence totale très supérieure à la latence de service | Limite de débit du palier gratuit Groq (~85 appels rapprochés) | Normal ; la gate porte sur la latence de service |
| `NotFoundError` à l'évaluation de la génération | Le modèle a été retiré par Groq (incident vécu le 2026-11-12 avec `llama-3.1-8b-instant`) | Choisir un modèle disponible dans `.env` (`GROQ_MODEL`), puis relancer `dvc repro` |
| Empreinte de `bitext.csv` différente | Le fichier a changé sur Hugging Face | Les étapes suivantes divergent : récupérer le fichier d'origine (voir limite ci-dessous) |
| Erreur `report_matches_params` dans les gates | `params.yaml` ou le prompt modifié sans relancer l'évaluation | Relancer `dvc repro`, puis committer les rapports |
| Avertissements `HF_TOKEN` ou `symlinks` (Windows) | Accès anonyme à Hugging Face, cache sans liens symboliques | Sans effet sur les résultats |

## 7. Ce qui est versionné, et où

| Élément | Versionné par | Emplacement |
|---|---|---|
| Code, tests, configuration, prompt | Git | dépôt GitHub |
| Données, index | DVC (empreintes dans `dvc.lock`, versionné par Git) | cache DVC local |
| Rapports d'évaluation et des gates | Git (métriques DVC non cachées) | `reports/` |
| Détail question par question | Git | `reports/generation_details.jsonl` |
| Runs d'expériences | MLflow | `mlflow.db` local (`mlflow ui --backend-store-uri sqlite:///mlflow.db`) |
| Secrets | jamais versionnés | `.env` local, modèle dans `.env.example` |

## 8. Limites de la reproductibilité

- **Pas de stockage DVC distant** : les données ne sont pas poussées (`dvc push`) ; elles sont
  régénérées depuis Hugging Face. Si le fichier source change, la reconstruction diverge,
  ce que l'empreinte MD5 permet de détecter. Amélioration : configurer un remote DVC
  (Google Drive, DagsHub, S3).
- **Dépendance à une API externe** : la génération exige une clé Groq, un accès réseau et
  la disponibilité du modèle.
- **Base MLflow locale** : l'historique des runs n'est pas partagé ; les résultats qui
  comptent sont aussi écrits dans `reports/`, versionné.
- **Dépendances indirectes non figées** : `requirements-lock.txt` fige les dépendances
  directes ; un `pip freeze` complet figerait aussi les dépendances indirectes.
