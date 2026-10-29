"""One self-contained HTML page with every table and figure the pipeline has produced so far."""
from __future__ import annotations

import base64
import html
from datetime import date
from pathlib import Path

import pandas as pd

from .config import Config

# (section title, note, [(table or figure name, caption)]); files that do not exist yet are listed as pending.
SECTIONS = [
    ("H1-A: FIRMS-only timing proxies",
     "Polar-orbit proxies for later ignition. Trends are Theil-Sen slopes with Mann-Kendall p over seasons.",
     [("h1a_proxy_panel.png", "Proxies by season, block-bootstrap 95% intervals"),
      ("h1a_proxy_trends.csv", "Trends"),
      ("h1a_aqua_dose_response.png", "Aqua / S-NPP ratio against Aqua's overpass hour"),
      ("h1a_aqua_dose_response.csv", "Aqua dose-response"),
      ("h1a_proxies_yearly.csv", "Proxy values per season"),
      ("overpass_times_ist.csv", "Median overpass hour per satellite (IST)")]),
    ("H1-B: geostationary ignition timing",
     "Needs LSA SAF MSG-IODC files.",
     [("h1b_ridgeline.png", "First-detection hour per season"),
      ("h1b_timing_yearly.csv", "Timing per season"),
      ("h1b_timing_trends.csv", "Trends")]),
    ("H2: share of fires missed by polar passes",
     "Needs geostationary events; estimator 3 needs Earth Engine.",
     [("h2_missed_share.png", "Missed share by estimator"),
      ("h2_missed_share.csv", "Missed share"),
      ("burned_vs_detected.png", "Burned area against detections")]),
    ("H3: decoupling of Delhi PM2.5 from detected fires",
     "log PM2.5 on year effects, year x lagged fire index, meteorology, Diwali, weekday and season spline; HAC errors.",
     [("h3_beta_by_year.png", "PM2.5 response per detected fire, by season"),
      ("h3_beta_trend.csv", "Trend in the per-season response"),
      ("h3_beta_by_year.csv", "Per-season response"),
      ("h3_counterfactual.png", "Observed minus predicted (model trained 2016-2021)"),
      ("h3_counterfactual_residuals.csv", "Counterfactual residuals"),
      ("h3_hidden_fire_test.csv", "Hidden-fire test: lambda 0 = detected counts explain PM2.5, 1 = fall in fires fully unmatched"),
      ("h3_functional_form.csv", "Fire term form (log1p / linear / sqrt): leave-one-season-out fit on training seasons"),
      ("h3_naive_yearly.csv", "Season totals, no controls (descriptive: excess per 1000 fires rises mechanically as fires fall)"),
      ("h3_lag_corr_heatmap.png", "Lagged correlations of daily changes"),
      ("robust_h3_variants.csv", "Robustness: alternative specifications"),
      ("robust_h3_loyo.csv", "Robustness: leave one season out")]),
    ("Robustness",
     "H1 checks; tracer substitution uses the US Embassy series when it exists.",
     [("h3_tracer_substitution.csv", "Tracer substitution (TROPOMI CO, MAIAC AOD)"),
      ("robust_drift_adjusted.csv", "H1-A trends after orbit-drift adjustment"),
      ("robust_h1_loyo.csv", "H1 leave-one-year-out")]),
]

# H3 runs on two PM2.5 series; the US Embassy run lives in outputs/usemb.
SERIES = [("usemb", "US Embassy station (main: continuous, all seasons)"),
          ("", "CPCB multi-station median (secondary: no 2022-2024 seasons)")]

CSS = """
:root { --bg:#fbfaf7; --fg:#1f2328; --muted:#5d636b; --line:#dcd8cf; --card:#ffffff; --accent:#9a4a1b; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --bg:#16181b; --fg:#e6e3dc; --muted:#a3a7ad; --line:#33373c; --card:#1e2125; --accent:#e39a62; } }
:root[data-theme="dark"] { --bg:#16181b; --fg:#e6e3dc; --muted:#a3a7ad; --line:#33373c; --card:#1e2125; --accent:#e39a62; }
* { box-sizing: border-box; }
body { margin:0; background:var(--bg); color:var(--fg); font:15px/1.55 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width:1040px; margin:0 auto; padding:32px 16px 64px; }
h1 { font-size:28px; margin:0 0 4px; } h2 { font-size:20px; margin:40px 0 4px; border-top:1px solid var(--line); padding-top:24px; }
h3 { font-size:15px; margin:20px 0 6px; } .muted { color:var(--muted); }
figure { margin:12px 0; background:var(--card); border:1px solid var(--line); border-radius:8px; padding:12px; }
figure img { max-width:100%; height:auto; display:block; margin:0 auto; background:#fff; border-radius:4px; }
figcaption { color:var(--muted); font-size:13px; margin-top:6px; }
.scroll { overflow-x:auto; border:1px solid var(--line); border-radius:8px; background:var(--card); }
table { border-collapse:collapse; font-size:13px; font-variant-numeric:tabular-nums; width:100%; }
th, td { padding:5px 10px; border-bottom:1px solid var(--line); text-align:right; white-space:nowrap; }
th:first-child, td:first-child { text-align:left; } th { color:var(--muted); font-weight:600; }
.pending { color:var(--muted); font-size:13px; margin:4px 0; } .pending::before { content:"pending: "; color:var(--accent); }
"""


def _table_html(path: Path) -> str:
    df = pd.read_csv(path)
    num = df.select_dtypes("number").columns
    df[num] = df[num].round(4)
    return f'<div class="scroll">{df.to_html(index=False, border=0, na_rep="")}</div>'


def _figure_html(path: Path, caption: str) -> str:
    data = base64.b64encode(path.read_bytes()).decode()
    return f'<figure><img src="data:image/png;base64,{data}" alt="{html.escape(caption)}"><figcaption>{html.escape(caption)}</figcaption></figure>'


def _item(out_root: Path, name: str, caption: str) -> tuple[str, bool]:
    path = out_root / ("figures" if name.endswith(".png") else "tables") / name
    if not path.exists():
        return f'<p class="pending">{html.escape(caption)} ({name})</p>', False
    if name.endswith(".png"):
        return _figure_html(path, caption), True
    return f"<h3>{html.escape(caption)}</h3>{_table_html(path)}", True


def build(cfg: Config) -> Path:
    parts, done, total = [], 0, 0
    for title, note, items in SECTIONS:
        parts.append(f"<h2>{html.escape(title)}</h2><p class='muted'>{html.escape(note)}</p>")
        h3 = title.startswith("H3")
        for sub, label in (SERIES if h3 else [("", "")]):
            root = cfg.outputs / sub if sub else cfg.outputs
            if h3:
                parts.append(f"<h3>PM2.5 series: {html.escape(label)}</h3>")
            for name, caption in items:
                block, ok = _item(root, name, caption)
                parts.append(block)
                done, total = done + ok, total + 1
    body = "\n".join(parts)
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>SFTA Results</title><style>{CSS}</style></head>
<body><main>
<h1>Stubble Fire Timing Analytics</h1>
<p class="muted">Punjab and Haryana stubble fires vs Delhi PM2.5, seasons {cfg.years[0]}-{cfg.years[-1]}.
Generated {date.today().isoformat()}; {done} of {total} outputs available.</p>
{body}
</main></body></html>"""
    out = cfg.outputs / "report.html"
    out.write_text(page, encoding="utf-8")
    print(f"Report: {out} ({done}/{total} outputs)")
    return out
