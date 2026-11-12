"""Rendu HTML de l'interface de chat (bulles, avatars, cartes de sources).

Sans dépendance à Streamlit : tout est testable. Sécurité : le texte produit par
le LLM est échappé AVANT la mise en forme (gras, listes, citations), donc une
réponse contenant du HTML ou un script injecté s'affiche comme du texte et ne
s'exécute jamais dans la page.

Les chaînes HTML sont écrites sans ligne vide ni indentation : Streamlit les
passe à son moteur Markdown, qui traiterait une ligne indentée comme du code.
"""

import html
import re

BRAND = "#0f6e6e"         # vert d'eau profond (identité ShopFès)
BRAND_SOFT = "#e2f0ee"    # bulle de l'assistant
USER_BG = "#f1f0ee"       # bulle de l'utilisateur
INK = "#1d1d1b"
MUTED = "#6b6a66"

CITE_RE = re.compile(r"\[(\d+)\]")
BULLET_RE = re.compile(r"^\s*(?:[-*•])\s+(.*)$")
NUMBER_RE = re.compile(r"^\s*\d+[.)]\s+(.*)$")

LOGO_SVG = (
    "<svg viewBox='0 0 40 40' width='40' height='40' aria-hidden='true'>"
    f"<rect width='40' height='40' rx='12' fill='{BRAND}'/>"
    "<path d='M12 16h16l-1.4 13.2a2 2 0 0 1-2 1.8h-9.2a2 2 0 0 1-2-1.8z' fill='#fff'/>"
    "<path d='M16 16v-2.5a4 4 0 0 1 8 0V16' fill='none' stroke='#fff' stroke-width='2.2' stroke-linecap='round'/>"
    "</svg>"
)

BOT_AVATAR = (
    "<svg viewBox='0 0 32 32' width='32' height='32' aria-hidden='true'>"
    f"<circle cx='16' cy='16' r='16' fill='{BRAND}'/>"
    "<path d='M10 13h12l-1 9.5a1.6 1.6 0 0 1-1.6 1.5h-6.8a1.6 1.6 0 0 1-1.6-1.5z' fill='#fff'/>"
    "<path d='M13 13v-1.5a3 3 0 0 1 6 0V13' fill='none' stroke='#fff' stroke-width='1.8' stroke-linecap='round'/>"
    "</svg>"
)

USER_AVATAR = (
    "<svg viewBox='0 0 32 32' width='32' height='32' aria-hidden='true'>"
    "<circle cx='16' cy='16' r='16' fill='#d9d7d1'/>"
    "<circle cx='16' cy='13' r='5' fill='#fff'/>"
    "<path d='M7 26a9 9 0 0 1 18 0' fill='#fff'/>"
    "</svg>"
)

CSS = f"""
#MainMenu, footer, header[data-testid="stHeader"] {{visibility:hidden;height:0}}
.block-container {{max-width:760px;padding-top:1.2rem;padding-bottom:6rem}}
.sf-header {{display:flex;align-items:center;gap:12px;padding:14px 18px;background:#fff;
border:1px solid #e6e4df;border-radius:18px;margin-bottom:18px}}
.sf-name {{font-weight:700;font-size:20px;color:{INK};line-height:1.2}}
.sf-status {{font-size:13px;color:{MUTED};display:flex;align-items:center;gap:6px}}
.sf-dot {{width:8px;height:8px;border-radius:50%;background:#1aa35a;display:inline-block}}
.sf-row {{display:flex;align-items:flex-end;gap:10px;margin:10px 0}}
.sf-row.user {{justify-content:flex-end}}
.sf-avatar {{flex:0 0 32px;height:32px}}
.sf-col {{display:flex;flex-direction:column;gap:6px;max-width:78%}}
.sf-row.user .sf-col {{align-items:flex-end}}
.sf-bubble {{padding:12px 16px;border-radius:20px;font-size:15.5px;line-height:1.5;color:{INK};
word-wrap:break-word}}
.sf-row.bot .sf-bubble {{background:{BRAND_SOFT};border-bottom-left-radius:6px}}
.sf-row.user .sf-bubble {{background:{USER_BG};border-bottom-right-radius:6px}}
.sf-row.bot.refused .sf-bubble {{background:#f4f2ee;border:1px dashed #cfccc4}}
.sf-bubble p {{margin:0 0 6px}} .sf-bubble p:last-child {{margin-bottom:0}}
.sf-bubble ul, .sf-bubble ol {{margin:4px 0 6px 18px;padding:0}}
.sf-cite {{display:inline-block;min-width:18px;padding:0 5px;margin-left:2px;border-radius:9px;
background:{BRAND};color:#fff;font-size:11px;font-weight:700;line-height:18px;text-align:center;
vertical-align:2px}}
.sf-card {{display:flex;align-items:center;gap:10px;padding:9px 12px;background:#fff;
border:1px solid #dfe9e7;border-radius:14px;font-size:13px;color:{INK};max-width:100%}}
.sf-card-icon {{flex:0 0 26px;height:26px;border-radius:8px;background:{BRAND_SOFT};display:grid;
place-items:center;font-size:13px;color:{BRAND};font-weight:700}}
.sf-card-title {{font-weight:600}}
.sf-card-sub {{color:{MUTED};font-size:12px}}
.sf-meta {{font-size:12px;color:{MUTED};padding-left:4px}}
.sf-tag {{display:inline-block;font-size:11.5px;color:{MUTED};background:#ecebe7;border-radius:8px;
padding:1px 8px;margin-bottom:2px}}
.sf-typing span {{display:inline-block;width:7px;height:7px;margin:0 2px;border-radius:50%;
background:{BRAND};opacity:.35;animation:sf-blink 1.2s infinite}}
.sf-typing span:nth-child(2) {{animation-delay:.2s}} .sf-typing span:nth-child(3) {{animation-delay:.4s}}
@keyframes sf-blink {{0%,80%,100% {{opacity:.25}} 40% {{opacity:1}}}}
.sf-note {{font-size:12px;color:{MUTED};text-align:center;margin:4px 0 10px}}
div.stButton > button {{border-radius:999px;border:1px solid #cfe2df;color:{BRAND};background:#fff;
font-size:14px;padding:4px 14px}}
div.stButton > button:hover {{border-color:{BRAND};color:{BRAND};background:{BRAND_SOFT}}}
[data-testid="stChatInput"] textarea {{font-size:15px}}
"""


def _inline(text: str) -> str:
    """Texte DÉJÀ échappé → gras, italique, citations [n] en pastilles."""
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"<em>\1</em>", text)
    return CITE_RE.sub(r"<span class='sf-cite'>\1</span>", text)


def safe_markdown(text: str) -> str:
    """Markdown simple (paragraphes, listes, gras, citations) → HTML sûr."""
    out, para, items, kind = [], [], [], None

    def flush_para():
        if para:
            out.append("<p>" + "<br>".join(para) + "</p>")
            para.clear()

    def flush_list():
        nonlocal kind
        if items:
            tag = "ul" if kind == "ul" else "ol"
            out.append(f"<{tag}>" + "".join(f"<li>{i}</li>" for i in items) + f"</{tag}>")
            items.clear()
        kind = None

    for raw in (text or "").splitlines():
        line = _inline(html.escape(raw.strip()))
        bullet, number = BULLET_RE.match(raw), NUMBER_RE.match(raw)
        if bullet or number:
            flush_para()
            new_kind = "ul" if bullet else "ol"
            if kind and kind != new_kind:
                flush_list()
            kind = new_kind
            items.append(_inline(html.escape((bullet or number).group(1).strip())))
        elif not raw.strip():
            flush_para()
            flush_list()
        else:
            flush_list()
            para.append(line)
    flush_para()
    flush_list()
    return "".join(out)


def header_html() -> str:
    return ("<div class='sf-header'>" + LOGO_SVG + "<div><div class='sf-name'>ShopFès</div>"
            "<div class='sf-status'><span class='sf-dot'></span>Assistant en ligne · retours, livraison, "
            "commandes</div></div></div>")


def source_card(source: dict) -> str:
    title = html.escape(str(source.get("title") or "Document"))
    origin = html.escape(str(source.get("source") or ""))
    sub = [origin]
    if source.get("version"):
        sub.append(f"version {html.escape(str(source['version']))}")
    if source.get("score") is not None:
        sub.append(f"pertinence {source['score']:.2f}")
    return (f"<div class='sf-card'><div class='sf-card-icon'>{int(source['ref'])}</div><div>"
            f"<div class='sf-card-title'>{title}</div><div class='sf-card-sub'>{' · '.join(sub)}</div>"
            "</div></div>")


def bubble(role: str, body_html: str, extras: str = "", refused: bool = False) -> str:
    """Une ligne de conversation : avatar + bulle (+ cartes et méta sous la bulle)."""
    if role == "user":
        return (f"<div class='sf-row user'><div class='sf-col'><div class='sf-bubble'>{body_html}</div>"
                f"</div><div class='sf-avatar'>{USER_AVATAR}</div></div>")
    cls = "sf-row bot refused" if refused else "sf-row bot"
    return (f"<div class='{cls}'><div class='sf-avatar'>{BOT_AVATAR}</div><div class='sf-col'>"
            f"<div class='sf-bubble'>{body_html}</div>{extras}</div></div>")


def user_bubble(question: str) -> str:
    return bubble("user", safe_markdown(question))


def typing_bubble() -> str:
    return bubble("bot", "<span class='sf-typing'><span></span><span></span><span></span></span>")


def assistant_bubble(answer: str, meta: dict | None = None, sources: list[dict] | None = None) -> str:
    meta = meta or {}
    refused = bool(meta.get("refused"))
    body = safe_markdown(answer)
    if refused and not meta.get("error"):
        body = "<span class='sf-tag'>Hors du périmètre de la documentation</span>" + body
    extras = ""
    if sources and not refused:
        extras += "".join(source_card(s) for s in sources)
    details = []
    if meta.get("latency"):
        details.append(html.escape(meta["latency"]))
    if meta.get("search_query"):
        details.append("recherche : « " + html.escape(meta["search_query"]) + " »")
    if meta.get("error"):
        details.append("erreur technique enregistrée : " + html.escape(str(meta["error"])))
    if details:
        extras += "<div class='sf-meta'>" + " · ".join(details) + "</div>"
    return bubble("bot", body, extras, refused)
