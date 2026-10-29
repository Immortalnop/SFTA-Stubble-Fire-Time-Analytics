"""Figures for deliverables 2, 4, 5 and 6."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import Config
from .proxies import HEADLINE, LABELS

plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False, "font.size": 9})
INK = "#1f4e79"


def _errorbars(ax, x, y, lo, hi, **kw):
    ax.errorbar(x, y, yerr=[np.asarray(y) - np.asarray(lo), np.asarray(hi) - np.asarray(y)],
                fmt="o-", color=INK, ecolor="#9db4cc", capsize=2, ms=4, lw=1.2, **kw)


def proxy_panel(cfg: Config, yearly: pd.DataFrame, trends: pd.DataFrame) -> None:
    names = [n for n in HEADLINE if n in set(yearly["proxy"])]
    fig, axes = plt.subplots(len(names), 1, figsize=(6.5, 1.9 * len(names)), sharex=True)
    axes = np.atleast_1d(axes)
    tr_all = trends.set_index("proxy")
    for ax, name in zip(axes, names):
        g = yearly[yearly["proxy"].eq(name)]
        _errorbars(ax, g["year"], g["value"], g["lo"], g["hi"])
        tr = tr_all.loc[name] if name in tr_all.index else None
        sub = f"Theil–Sen {tr['slope']:+.3g}/yr, MK p={tr['mk_p']:.2g}" if tr is not None and np.isfinite(tr["slope"]) else ""
        ax.set_title(f"{LABELS[name]}   {sub}", loc="left", fontsize=8.5)
    axes[-1].set_xlabel("Season (Sep 15 – Nov 30); 2026 partial")
    fig.tight_layout()
    fig.savefig(cfg.figure("h1a_proxy_panel.png"))
    plt.close(fig)


def aqua_dose_response(cfg: Config, dr: pd.DataFrame) -> None:
    """Aqua/S-NPP ratio against Aqua's median overpass hour, relative to the pre-drift level."""
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    ax.plot(dr["aqua_hour"], dr["relative"], "o", color=INK)
    for _, r in dr.iterrows():
        ax.annotate(int(r["year"]), (r["aqua_hour"], r["relative"]), textcoords="offset points", xytext=(4, 4), fontsize=7)
    ax.axhline(1, color="#999", lw=0.8)
    ax.set_xlabel("Aqua median overpass over the region (IST hour)")
    ax.set_ylabel("Aqua/S-NPP, relative to 2016–2022")
    ax.set_title("Fire activity seen by a later-drifting sensor", loc="left", fontsize=9)
    fig.tight_layout()
    fig.savefig(cfg.figure("h1a_aqua_dose_response.png"))
    plt.close(fig)


def corr_heatmap(cfg: Config, corr: pd.DataFrame) -> None:
    m = corr.pivot(index="year", columns="lag", values="r")
    fig, ax = plt.subplots(figsize=(4.5, 0.35 * len(m) + 1.2))
    im = ax.imshow(m.to_numpy(), cmap="RdBu_r", vmin=-0.6, vmax=0.6, aspect="auto")
    ax.set_xticks(range(m.shape[1]), [f"{c} d" for c in m.columns])
    ax.set_yticks(range(m.shape[0]), m.index)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            v = m.iat[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7)
    ax.set_xlabel("Fire lead (days)")
    ax.set_title("Δlog fires vs Δlog Delhi PM2.5", loc="left")
    fig.colorbar(im, ax=ax, shrink=0.8, label="r")
    fig.tight_layout()
    fig.savefig(cfg.figure("h3_lag_corr_heatmap.png"))
    plt.close(fig)


def beta_plot(cfg: Config, beta: pd.DataFrame, trend: dict) -> None:
    fig, ax = plt.subplots(figsize=(6, 3))
    _errorbars(ax, beta["year"], beta["beta"], beta["lo"], beta["hi"])
    ax.axhline(0, color="#999", lw=0.8)
    ax.set_ylabel("β: Δlog PM2.5 per Δlog(1+fire index)")
    ax.set_title(f"Per-fire PM2.5 response by season — slope {trend['beta_slope_per_year']:+.3f}/yr "
                 f"(p={trend['slope_p']:.2f})", loc="left", fontsize=9)
    fig.tight_layout()
    fig.savefig(cfg.figure("h3_beta_by_year.png"))
    plt.close(fig)


def counterfactual_plot(cfg: Config, resid: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6, 3))
    tr = resid[resid["train"]]
    te = resid[~resid["train"]]
    ax.axhline(0, color="#999", lw=0.8)
    ax.errorbar(tr["year"], tr["mean_resid_log"], yerr=[tr["mean_resid_log"] - tr["lo"], tr["hi"] - tr["mean_resid_log"]],
                fmt="o", color="#9db4cc", capsize=2, label="training seasons")
    ax.errorbar(te["year"], te["mean_resid_log"], yerr=[te["mean_resid_log"] - te["lo"], te["hi"] - te["mean_resid_log"]],
                fmt="o", color=INK, capsize=2, label="predicted seasons")
    ax.set_ylabel("Mean log(observed / predicted)")
    ax.set_title("PM2.5 left unexplained by detected fires + met", loc="left")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(cfg.figure("h3_counterfactual.png"))
    plt.close(fig)


# ---------------------------------------------------------------- H1-B / H2 / burned area

def ridgeline(cfg: Config, ev: pd.DataFrame, weights: np.ndarray | None = None) -> None:
    """Deliverable 1: density of GEO event first-detection time (IST) by year."""
    from scipy.stats import gaussian_kde

    years = sorted(ev["year"].unique())
    grid = np.linspace(6, 22, 321)
    fig, ax = plt.subplots(figsize=(6.5, 0.55 * len(years) + 1.5))
    w_all = pd.Series(weights if weights is not None else np.ones(len(ev)), index=ev.index)
    for i, y in enumerate(years):
        sel = ev["year"].eq(y)
        h = ev.loc[sel, "hour_ist"].to_numpy()
        if len(h) < 5:
            continue
        dens = gaussian_kde(h, weights=w_all[sel].to_numpy(), bw_method=0.25)(grid)
        dens = dens / dens.max() * 0.9
        ax.fill_between(grid, i, i + dens, color=INK, alpha=0.75, lw=0, zorder=len(years) - i)
        ax.plot(grid, i + dens, color="white", lw=0.8, zorder=len(years) - i)
        med = np.median(h)
        ax.plot([med, med], [i, i + 0.9], color="#e07b00", lw=1.2, zorder=len(years) - i + 1)
    ax.axvline(13.5, color="#999", lw=0.8, ls=":")
    ax.text(13.55, len(years) + 0.1, "13:30 pass", fontsize=7, color="#666")
    ax.set_yticks([i + 0.2 for i in range(len(years))], years)
    ax.set_xlabel("Event first detection, hour of day (IST)")
    ax.set_title("When fires start: Meteosat IODC, orange = yearly median", loc="left", fontsize=9)
    ax.set_xlim(6, 22)
    fig.tight_layout()
    fig.savefig(cfg.figure("h1b_ridgeline.png"))
    plt.close(fig)


def h2_panel(cfg: Config, table: pd.DataFrame) -> None:
    """Deliverable 3 as a figure: three estimators per year."""
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    spec = [("est1_share", "est1_lo", "est1_hi", "GEO share after last pass", INK),
            ("est2_missed", "est2_lo", "est2_hi", "Detectability model: P(missed)", "#e07b00")]
    for k, (v, lo, hi, label, col) in enumerate(spec):
        d = table.dropna(subset=[v])
        ax.errorbar(d["year"] + (k - 0.5) * 0.15, d[v], yerr=[d[v] - d[lo], d[hi] - d[v]], fmt="o-", color=col,
                    capsize=2, ms=4, lw=1, label=label)
    if "est3_missed_upper" in table and table["est3_missed_upper"].notna().any():
        d = table.dropna(subset=["est3_missed_upper"])
        ax.plot(d["year"] + 0.15, d["est3_missed_upper"], "s--", color="#4c8f4c", ms=4, lw=1, label="Burned area not near a VIIRS hit (upper bound)")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Share of fires missed by polar overpasses")
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(cfg.figure("h2_missed_share.png"))
    plt.close(fig)


def burned_vs_detected(cfg: Config, ba: pd.DataFrame) -> None:
    """Deliverable 7: burned area and VIIRS detection count, each indexed to its own peak year (=100)."""
    agg = ba.groupby("year")[["total_ha", "n_viirs"]].sum()
    idx = 100 * agg / agg.max()
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(idx.index, idx["total_ha"], "o-", color="#4c8f4c", label="Burned area (Sentinel-2 dNBR)")
    ax.plot(idx.index, idx["n_viirs"], "o-", color=INK, label="VIIRS detections")
    ax.set_ylabel("Index (peak year = 100)")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Sample districts: burned area vs detected fires", loc="left", fontsize=9)
    fig.tight_layout()
    fig.savefig(cfg.figure("burned_vs_detected.png"))
    plt.close(fig)
