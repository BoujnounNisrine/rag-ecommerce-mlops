"""Interface de chat de l'assistant ShopFès (bulles, avatars, cartes de sources).

Lancement, depuis la racine du projet :
    streamlit run src/app/streamlit_app.py

Réutilise exactement le pipeline de la ligne de commande (traduction, retrieval,
refus, génération Groq, citations). Chaque question est enregistrée dans
logs/requests.jsonl (sans son texte) et apparaît dans le tableau de bord de
monitoring. L'historique reste dans la session du navigateur, jamais sur le disque.
"""

import sys
from pathlib import Path

# Streamlit exécute ce fichier comme un script : on rend le paquet `src` importable.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from src.app.chat_ui import CSS, assistant_bubble, header_html, typing_bubble, user_bubble  # noqa: E402
from src.app.formatting import DISCLAIMER, EXAMPLES, response_meta  # noqa: E402
from src.config import load_params  # noqa: E402
from src.rag.prompt import PROMPT_VERSION  # noqa: E402

WELCOME = (
    "Bonjour ! Je suis l'assistant de ShopFès. Je réponds à vos questions sur les **retours**, "
    "la **livraison**, vos **commandes** et votre **compte**, à partir de notre documentation."
)

st.set_page_config(page_title="Assistant ShopFès", page_icon="🛍️", layout="centered",
                   initial_sidebar_state="collapsed")


def html_block(markup: str) -> None:
    st.markdown(markup, unsafe_allow_html=True)


@st.cache_resource(show_spinner="Chargement du modèle et de l'index…")
def get_pipeline():
    from src.rag.ask import build_pipeline

    return build_pipeline()


html_block(f"<style>{CSS}</style>")

# Barre latérale (repliée par défaut) : configuration et remise à zéro ----------------

params = load_params()
chunking, retrieval = params["chunking"], params["retrieval"]
with st.sidebar:
    st.subheader("Configuration active")
    st.markdown(
        f"- Embeddings : `{params['embeddings']['model'].split('/')[-1]}`\n"
        f"- Chunks : {chunking['chunk_size']} mots, chevauchement {chunking['chunk_overlap']}\n"
        f"- Passages par réponse : {retrieval['top_k']}\n"
        f"- Seuil de refus : {retrieval['min_score']}\n"
        f"- Prompt : {PROMPT_VERSION}\n"
        f"- Traduction de la requête : {'oui' if retrieval.get('translate_query') else 'non'}"
    )
    if st.button("Nouvelle conversation"):
        st.session_state.messages = []
        st.rerun()

# Conversation --------------------------------------------------------------------

html_block(header_html())

try:
    pipeline = get_pipeline()
except FileNotFoundError:
    st.error("Index introuvable. Lancez d'abord `dvc repro` depuis la racine du projet.")
    st.stop()
except RuntimeError as exc:  # clé Groq manquante
    st.error(str(exc))
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

html_block(assistant_bubble(WELCOME))
html_block(f"<div class='sf-note'>{DISCLAIMER}</div>")

for m in st.session_state.messages:
    if m["role"] == "user":
        html_block(user_bubble(m["content"]))
    else:
        html_block(assistant_bubble(m["content"], m["meta"], m["sources"]))

typed = st.chat_input("Écrivez votre message…")
question = typed or st.session_state.pop("pending", None)

if question:
    html_block(user_bubble(question))
    slot = st.empty()
    with slot.container():
        html_block(typing_bubble())
    resp = pipeline.ask(question)
    meta = response_meta(resp, question)
    slot.empty()
    html_block(assistant_bubble(resp.answer, meta, resp.sources))
    st.session_state.messages += [
        {"role": "user", "content": question},
        {"role": "assistant", "content": resp.answer, "meta": meta, "sources": resp.sources},
    ]

# Suggestions cliquables, tant que la conversation n'a pas commencé
if not st.session_state.messages:
    st.caption("Suggestions")
    cols = st.columns(2)
    for i, example in enumerate(EXAMPLES):
        if cols[i % 2].button(example, key=f"ex-{i}"):
            st.session_state.pending = example
            st.rerun()
