from src.app.chat_ui import (
    assistant_bubble,
    header_html,
    safe_markdown,
    source_card,
    typing_bubble,
    user_bubble,
)

SRC = {"ref": 1, "chunk_id": "p#0", "source": "policies/return_policy.md", "title": "Return window",
       "version": "2026-09", "score": 0.69}


def test_llm_html_is_escaped_never_executed():
    out = safe_markdown("Voici <script>alert('x')</script> et <img src=x onerror=alert(1)>")
    assert "<script>" not in out and "<img" not in out
    assert "&lt;script&gt;" in out


def test_bold_citations_and_paragraphs():
    out = safe_markdown("Le délai est de **30 jours** [1][2].\n\nMerci.")
    assert "<strong>30 jours</strong>" in out
    assert out.count("class='sf-cite'") == 2
    assert out.count("<p>") == 2


def test_bullet_and_numbered_lists():
    out = safe_markdown("Étapes :\n1. Connectez-vous\n2. Ouvrez l'historique\n- carte cadeau\n- logiciel")
    assert "<ol><li>Connectez-vous</li><li>Ouvrez l&#x27;historique</li></ol>" in out
    assert "<ul><li>carte cadeau</li><li>logiciel</li></ul>" in out


def test_output_has_no_blank_line_or_indent_for_streamlit_markdown():
    for markup in (header_html(), typing_bubble(), user_bubble("bonjour"),
                   assistant_bubble("Réponse [1].", {"latency": "1.2 s"}, [SRC])):
        assert "\n" not in markup


def test_source_card_is_escaped():
    card = source_card({**SRC, "title": "<b>piège</b>"})
    assert "&lt;b&gt;piège&lt;/b&gt;" in card and "version 2026-09" in card and "pertinence 0.69" in card


def test_answer_shows_source_cards_and_meta():
    out = assistant_bubble("30 jours [1].", {"latency": "1.8 s", "search_query": "return window"}, [SRC])
    assert "sf-card" in out and "Return window" in out and "recherche : « return window »" in out


def test_refusal_hides_sources_and_is_tagged():
    out = assistant_bubble("Désolé…", {"refused": True, "latency": "0.3 s"}, [SRC])
    assert "refused" in out and "Hors du périmètre" in out and "sf-card" not in out


def test_user_and_bot_sides():
    assert "sf-row user" in user_bubble("question")
    assert "sf-row bot" in assistant_bubble("réponse")
