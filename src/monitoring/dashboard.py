"""Tableau de bord de monitoring : logs/requests.jsonl -> reports/monitoring.html.

Page HTML autonome (graphiques SVG intégrés, aucune dépendance réseau) :
  - tuiles : requêtes, erreurs, latence de service P95, refus, coût ;
  - alertes avec seuils (params.yaml, section monitoring) ;
  - latence par requête : service vs totale (l'écart = attente due à la limite de débit) ;
  - distribution du meilleur score de retrieval, avec le seuil de refus ;
  - activité par heure et dérive des entrées (fenêtre récente vs précédente).

Usage :
  python -m src.monitoring.dashboard
puis ouvrir reports/monitoring.html dans un navigateur.
"""

import json
from datetime import datetime, timezone
from html import escape

from src.config import load_params, resolve
from src.data.io import read_jsonl
from src.monitoring.metrics import alerts, by_hour, drift, summarize

W, H = 720, 260
PAD_L, PAD_R, PAD_T, PAD_B = 56, 84, 16, 36

CSS = """
.viz-root{color-scheme:light;--surface-0:#f5f4f1;--surface-1:#fcfcfb;--text-primary:#0b0b0b;
--text-secondary:#52514e;--text-muted:#7a7974;--grid:#e6e5e0;--axis:#c9c8c2;
--series-1:#2a78d6;--series-2:#eb6834;--good:#0ca30c;--critical:#d03b3b;--warning:#fab219}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme="light"])) .viz-root{color-scheme:dark;
--surface-0:#111110;--surface-1:#1a1a19;--text-primary:#f0efec;--text-secondary:#c3c2b7;--text-muted:#94938b;
--grid:#2c2c2a;--axis:#4a4a46;--series-1:#3987e5;--series-2:#d95926}}
:root[data-theme="dark"] .viz-root{color-scheme:dark;--surface-0:#111110;--surface-1:#1a1a19;
--text-primary:#f0efec;--text-secondary:#c3c2b7;--text-muted:#94938b;--grid:#2c2c2a;--axis:#4a4a46;
--series-1:#3987e5;--series-2:#d95926}
*{box-sizing:border-box}
body{margin:0;font:15px/1.5 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
.viz-root{background:var(--surface-0);color:var(--text-primary);min-height:100vh}
main{max-width:980px;margin:0 auto;padding:24px 16px 48px}
h1{font-size:24px;margin:0 0 4px}h2{font-size:17px;margin:0 0 4px}
.sub{color:var(--text-secondary);margin:0 0 20px}
.note{color:var(--text-muted);font-size:13px;margin:4px 0 12px}
.card{background:var(--surface-1);border:1px solid var(--grid);border-radius:10px;padding:16px;margin:16px 0}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px}
.tile{background:var(--surface-1);border:1px solid var(--grid);border-radius:10px;padding:12px 14px}
.tile .label{color:var(--text-secondary);font-size:13px}
.tile .value{font-size:26px;font-weight:600;font-variant-numeric:tabular-nums;margin-top:2px}
.tile .detail{color:var(--text-muted);font-size:12px}
.status{display:inline-flex;align-items:center;gap:6px;font-size:13px;font-weight:600}
.status .dot{width:16px;height:16px;border-radius:50%;display:inline-grid;place-items:center;
color:#fff;font-size:11px;line-height:1}
.ok .dot{background:var(--good)}.alert .dot{background:var(--critical)}.na .dot{background:var(--text-muted)}
ul.alerts{list-style:none;padding:0;margin:8px 0 0}
ul.alerts li{display:flex;justify-content:space-between;gap:12px;padding:8px 0;border-top:1px solid var(--grid);
flex-wrap:wrap}
ul.alerts li:first-child{border-top:0}
.mono{font-variant-numeric:tabular-nums;color:var(--text-secondary)}
.legend{display:flex;gap:16px;flex-wrap:wrap;font-size:13px;color:var(--text-secondary);margin:6px 0}
.legend span{display:inline-flex;align-items:center;gap:6px}
.swatch{width:14px;height:3px;border-radius:2px;display:inline-block}
svg{width:100%;height:auto;display:block}
svg text{fill:var(--text-muted);font-size:11px}
svg .lbl{fill:var(--text-secondary);font-size:12px}
svg .hit{fill:transparent}svg .hit:hover{fill:var(--text-muted);fill-opacity:.15}
table{width:100%;border-collapse:collapse;font-size:14px;font-variant-numeric:tabular-nums}
th,td{text-align:right;padding:6px 8px;border-top:1px solid var(--grid)}
th:first-child,td:first-child{text-align:left}
th{color:var(--text-secondary);font-weight:600;border-top:0}
.tablewrap{overflow-x:auto}
details summary{cursor:pointer;color:var(--text-secondary);font-size:14px;margin-top:8px}
"""


def _fmt(v, unit="", digits=2):
    if v is None:
        return "n/a"
    if unit == "%":
        return f"{v * 100:.1f} %"
    if unit == "s":
        return f"{v / 1000:.{digits}f} s"
    if unit == "$":
        return f"{v:.4f} $"
    return f"{v}{unit}"


def _status(state: str) -> str:
    icon, label, cls = {"ok": ("✓", "OK", "ok"), "alert": ("!", "Alerte", "alert"),
                        "n/a": ("–", "Non mesurable", "na")}[state]
    return f'<span class="status {cls}"><span class="dot" aria-hidden="true">{icon}</span>{label}</span>'


def _tile(label: str, value: str, detail: str = "", state: str | None = None) -> str:
    status = f"<div>{_status(state)}</div>" if state else ""
    return (f'<div class="tile"><div class="label">{escape(label)}</div>'
            f'<div class="value">{escape(value)}</div><div class="detail">{escape(detail)}</div>{status}</div>')


def _nice_max(v: float) -> float:
    for step in (1000, 2000, 5000, 10000, 20000, 30000, 60000, 120000):
        if v <= step:
            return step
    return v


def latency_chart(records: list[dict], threshold_ms: float) -> str:
    # s = latence de service, None pour les requêtes antérieures à l'instrumentation
    pts = [(i, (r.get("latency_ms") or {}).get("service"), (r.get("latency_ms") or {}).get("total"))
           for i, r in enumerate(records)]
    pts = [(i, s, t) for i, s, t in pts if t is not None]
    if len(pts) < 2:
        return '<p class="note">Pas assez de requêtes pour tracer la latence.</p>'
    ymax = _nice_max(max(max(t for _, _, t in pts), threshold_ms) * 1.05)
    n = len(pts)
    pw, ph = W - PAD_L - PAD_R, H - PAD_T - PAD_B

    def x(i):
        return PAD_L + pw * i / (n - 1)

    def y(v):
        return PAD_T + ph * (1 - v / ymax)

    parts = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Latence par requête, service et totale">']
    for k in range(5):  # grille horizontale
        v = ymax * k / 4
        parts.append(f'<line x1="{PAD_L}" x2="{W - PAD_R}" y1="{y(v):.1f}" y2="{y(v):.1f}" '
                     f'stroke="var(--grid)" stroke-width="1"/>')
        parts.append(f'<text x="{PAD_L - 8}" y="{y(v) + 4:.1f}" text-anchor="end">{v / 1000:g} s</text>')
    parts.append(f'<line x1="{PAD_L}" x2="{W - PAD_R}" y1="{y(0):.1f}" y2="{y(0):.1f}" stroke="var(--axis)"/>')
    parts.append(f'<text x="{PAD_L}" y="{H - 10}">requête 1</text>')
    parts.append(f'<text x="{W - PAD_R}" y="{H - 10}" text-anchor="end">requête {n}</text>')
    ty = y(threshold_ms)
    parts.append(f'<line x1="{PAD_L}" x2="{W - PAD_R}" y1="{ty:.1f}" y2="{ty:.1f}" stroke="var(--text-muted)" '
                 f'stroke-width="1" stroke-dasharray="4 4"/>')
    parts.append(f'<text class="lbl" x="{W - PAD_R + 6}" y="{ty + 4:.1f}">seuil {threshold_ms / 1000:g} s</text>')
    # Zone grisée : requêtes non instrumentées (service non mesuré)
    first_instr = next((k for k, p in enumerate(pts) if p[1] is not None), None)
    shade_end = x(first_instr) if first_instr is not None else W - PAD_R
    if first_instr != 0:
        parts.append(f'<rect x="{PAD_L}" y="{PAD_T}" width="{shade_end - PAD_L:.1f}" height="{ph}" '
                     f'fill="var(--grid)" fill-opacity=".5"/>')
        parts.append(f'<text x="{PAD_L + 6}" y="{PAD_T + 14}">avant instrumentation : service non mesuré</text>')
    # Courbe totale continue ; courbe de service seulement là où elle est mesurée
    total_path = " ".join(f"{'M' if k == 0 else 'L'}{x(k):.1f},{y(p[2]):.1f}" for k, p in enumerate(pts))
    service_path, pen_down = [], False
    for k, p in enumerate(pts):
        if p[1] is None:
            pen_down = False
            continue
        service_path.append(f"{'L' if pen_down else 'M'}{x(k):.1f},{y(p[1]):.1f}")
        pen_down = True
    for path, color in ((total_path, "var(--series-2)"), (" ".join(service_path), "var(--series-1)")):
        if path:
            parts.append(f'<path d="{path}" fill="none" stroke="{color}" stroke-width="2" '
                         f'stroke-linejoin="round" stroke-linecap="round"/>')
    last = pts[-1]  # étiquettes directes en fin de courbe
    yt = y(last[2])
    if last[1] is not None:
        ys = y(last[1])
        if abs(ys - yt) < 14:
            yt = ys - 14
        parts.append(f'<text class="lbl" x="{W - PAD_R + 6}" y="{ys + 4:.1f}">service</text>')
    if abs(yt - ty) < 14:  # ne pas chevaucher l'étiquette du seuil
        yt = ty + 14 if yt >= ty else ty - 14
    parts.append(f'<text class="lbl" x="{W - PAD_R + 6}" y="{yt + 4:.1f}">totale</text>')
    band = pw / max(n - 1, 1)
    for k, (_, s, t) in enumerate(pts):  # zones de survol (plus larges que les points)
        detail = (f"service {s / 1000:.2f} s, totale {t / 1000:.2f} s, attente débit {(t - s) / 1000:.2f} s"
                  if s is not None else f"totale {t / 1000:.2f} s (service non mesuré)")
        parts.append(f'<rect class="hit" x="{x(k) - band / 2:.1f}" y="{PAD_T}" width="{max(band, 2):.1f}" '
                     f'height="{ph}"><title>Requête {k + 1} — {detail}</title></rect>')
    parts.append("</svg>")
    return "".join(parts)


def score_histogram(records: list[dict], min_score: float, bins: int = 20) -> str:
    scores = [r["top_score"] for r in records if r.get("top_score") is not None]
    if not scores:
        return '<p class="note">Aucun score de retrieval enregistré.</p>'
    counts = [0] * bins
    for s in scores:
        counts[min(max(int(s * bins), 0), bins - 1)] += 1
    step = next(s for s in (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 10**9) if max(counts) / s <= 5)
    cmax = step * -(-max(counts) // step)  # graduations entières : 0, step, 2·step…
    pw, ph = W - PAD_L - PAD_R, H - PAD_T - PAD_B
    bw = pw / bins

    def y(c):
        return PAD_T + ph * (1 - c / cmax)

    parts = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Distribution du meilleur score de retrieval">']
    for c in range(0, cmax + 1, step):
        parts.append(f'<line x1="{PAD_L}" x2="{W - PAD_R}" y1="{y(c):.1f}" y2="{y(c):.1f}" '
                     f'stroke="var(--grid)"/><text x="{PAD_L - 8}" y="{y(c) + 4:.1f}" '
                     f'text-anchor="end">{c:.0f}</text>')
    for b, c in enumerate(counts):
        if c == 0:
            continue
        x0 = PAD_L + b * bw + 1  # écart de 2 px entre barres
        top, h = y(c), PAD_T + ph - y(c)
        r = min(4, bw / 2 - 1, h)
        parts.append(f'<path d="M{x0:.1f},{PAD_T + ph:.1f} V{top + r:.1f} Q{x0:.1f},{top:.1f} {x0 + r:.1f},{top:.1f} '
                     f'H{x0 + bw - 2 - r:.1f} Q{x0 + bw - 2:.1f},{top:.1f} {x0 + bw - 2:.1f},{top + r:.1f} '
                     f'V{PAD_T + ph:.1f} Z" fill="var(--series-1)"><title>Score {b / bins:.2f}–'
                     f'{(b + 1) / bins:.2f} : {c} requête(s)</title></path>')
    parts.append(f'<line x1="{PAD_L}" x2="{W - PAD_R}" y1="{PAD_T + ph}" y2="{PAD_T + ph}" stroke="var(--axis)"/>')
    for t in (0, 0.25, 0.5, 0.75, 1.0):
        parts.append(f'<text x="{PAD_L + pw * t:.1f}" y="{H - 14}" text-anchor="middle">{t:g}</text>')
    tx = PAD_L + pw * min_score
    parts.append(f'<line x1="{tx:.1f}" x2="{tx:.1f}" y1="{PAD_T}" y2="{PAD_T + ph}" stroke="var(--text-muted)" '
                 f'stroke-dasharray="4 4"/><text class="lbl" x="{tx + 6:.1f}" y="{PAD_T + 12}">'
                 f'seuil de refus {min_score:g}</text>')
    parts.append("</svg>")
    return "".join(parts)


def render(summary: dict, hours: list[dict], drift_info: dict, alert_list: list[dict],
           records: list[dict], params: dict) -> str:
    state = {a["name"]: a["status"] for a in alert_list}
    n_alerts = sum(a["status"] == "alert" for a in alert_list)
    threshold = params["monitoring"]["alerts"]["latency_service_p95_max_ms"]
    tiles = "".join([
        _tile("Requêtes", str(summary["n_requests"]), f"{summary.get('tokens_total', 0):,} tokens".replace(",", " ")),
        _tile("Taux d'erreur", _fmt(summary.get("error_rate"), "%"), "erreurs techniques",
              state.get("Taux d'erreur")),
        _tile("Latence de service P95", _fmt(summary.get("latency_service_p95_ms"), "s"),
              f"P50 {_fmt(summary.get('latency_service_p50_ms'), 's')} · "
              f"sur {summary.get('n_instrumented', 0)} requêtes instrumentées · "
              f"totale P95 {_fmt(summary.get('latency_total_p95_ms'), 's')} (toutes)",
              state.get("Latence de service P95 (ms)")),
        _tile("Taux de refus", _fmt(summary.get("refusal_rate"), "%"), "inclut les questions de test hors périmètre"),
        _tile("Coût estimé / 1000 req.", _fmt(summary.get("cost_per_1000_requests_usd"), "$"),
              f"{summary.get('tokens_per_request_mean')} tokens / requête"),
    ])
    alert_items = "".join(
        f'<li><span>{escape(a["name"])}</span><span class="mono">{_fmt(a["value"])} '
        f'(seuil {a["threshold"]})</span>{_status(a["status"])}</li>' for a in alert_list)
    hour_rows = "".join(
        f'<tr><td>{escape(h["hour"])}</td><td>{h["requests"]}</td><td>{h["errors"]}</td><td>{h["refusals"]}</td>'
        f'<td>{_fmt(h["latency_service_p95_ms"], "s")}</td><td>{h["tokens"]}</td></tr>' for h in hours)
    if drift_info.get("available"):
        drift_html = (f'<p>Fenêtre de {drift_info["window"]} requêtes, comparée à la précédente. '
                      f'Score moyen de retrieval : <b>{drift_info["top_score_ref"]}</b> → '
                      f'<b>{drift_info["top_score_current"]}</b>. Longueur moyenne des questions : '
                      f'<b>{drift_info["question_length_ref"]}</b> → <b>{drift_info["question_length_current"]}</b> '
                      f'caractères.</p>')
    else:
        drift_html = f'<p class="note">Dérive non calculable : {escape(drift_info.get("reason", ""))}.</p>'
    recent = records[-30:]
    lat_rows = "".join(
        f'<tr><td>{escape((r.get("timestamp") or "")[11:19])}</td>'
        f'<td>{_fmt((r.get("latency_ms") or {}).get("service"), "s")}</td>'
        f'<td>{_fmt((r.get("latency_ms") or {}).get("total"), "s")}</td>'
        f'<td>{"oui" if r.get("refused") else "non"}</td><td>{escape(str(r.get("error") or "—"))}</td>'
        f'<td>{r.get("top_score")}</td></tr>' for r in recent)
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")  # noqa: UP017
    period = f'{(summary.get("first") or "")[:16].replace("T", " ")} → {(summary.get("last") or "")[:16].replace("T", " ")}'
    headline = (f'{n_alerts} alerte(s) active(s)' if n_alerts else "Aucune alerte active")

    return f"""<!doctype html>
<html lang="fr"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Monitoring ShopFès</title><style>{CSS}</style></head>
<body><div class="viz-root"><main>
<h1>Monitoring de l'assistant ShopFès</h1>
<p class="sub">Période (UTC) : {escape(period)} · généré le {generated} · {headline}</p>
<section class="tiles">{tiles}</section>

<section class="card"><h2>Alertes</h2>
<p class="note">Seuils définis dans <code>params.yaml</code> (section monitoring), alignés sur les quality gates.</p>
<ul class="alerts">{alert_items}</ul></section>

<section class="card"><h2>Latence par requête</h2>
<p class="note">L'écart entre les deux courbes est l'attente imposée par la limite de débit de l'API
(palier gratuit), pas du temps de calcul. Survolez la courbe pour le détail.</p>
<div class="legend"><span><i class="swatch" style="background:var(--series-1)"></i>Latence de service</span>
<span><i class="swatch" style="background:var(--series-2)"></i>Latence totale</span></div>
{latency_chart(records, threshold)}
<details><summary>Voir les 30 dernières requêtes sous forme de tableau</summary>
<div class="tablewrap"><table><thead><tr><th>Heure (UTC)</th><th>Service</th><th>Totale</th><th>Refus</th>
<th>Erreur</th><th>Meilleur score</th></tr></thead><tbody>{lat_rows}</tbody></table></div></details></section>

<section class="card"><h2>Meilleur score de retrieval</h2>
<p class="note">Sous le seuil, la question est refusée sans appel au LLM. Un glissement de la distribution
vers la gauche signale des questions qui s'éloignent du corpus (dérive).</p>
{score_histogram(records, params["retrieval"]["min_score"])}</section>

<section class="card"><h2>Dérive des entrées</h2>{drift_html}</section>

<section class="card"><h2>Activité par heure (UTC)</h2><div class="tablewrap"><table>
<thead><tr><th>Heure</th><th>Requêtes</th><th>Erreurs</th><th>Refus</th><th>Service P95</th><th>Tokens</th></tr></thead>
<tbody>{hour_rows}</tbody></table></div></section>

<p class="note">Sources : logs/requests.jsonl (sans le texte des questions). Qualité des réponses mesurée hors ligne :
reports/generation_metrics.json.</p>
</main></div></body></html>"""


def main() -> None:
    params = load_params()
    log_path = resolve(params["data"]["request_log"])
    if not log_path.exists():
        print(f"Aucun journal trouvé ({log_path}). Posez des questions avec "
              "python -m src.rag.ask ou lancez l'évaluation de la génération.")
        return
    records = read_jsonl(log_path)
    g, m = params["generation"], params["monitoring"]
    summary = summarize(records, g["price_per_million_input"], g["price_per_million_output"])
    hours = by_hour(records)
    drift_info = drift(records, m["window"])
    alert_list = alerts(summary, drift_info, m["alerts"])

    out_html = resolve("reports/monitoring.html")
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(render(summary, hours, drift_info, alert_list, records, params), encoding="utf-8")
    resolve("reports/monitoring.json").write_text(json.dumps(
        {"summary": summary, "drift": drift_info, "alerts": alert_list, "by_hour": hours},
        indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"{summary['n_requests']} requêtes analysées.")
    for a in alert_list:
        mark = {"ok": "OK    ", "alert": "ALERTE", "n/a": "n/a   "}[a["status"]]
        print(f"  [{mark}] {a['name']:48s} {a['value']} (seuil {a['threshold']})")
    print(f"\nTableau de bord : {out_html}")


if __name__ == "__main__":
    main()
