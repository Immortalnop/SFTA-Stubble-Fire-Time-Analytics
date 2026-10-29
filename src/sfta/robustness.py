"""Robustness (plan section 7): leave-one-year-out, alternative specifications, orbit-drift adjustment.

Not implemented: UP/MP spillover (needs a second FIRMS download over a wider box) and a GEO cloud-fraction
covariate (needs the QualityProduct layout; see `python -m sfta geo-inspect`).
"""
from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd

from . import analysis
from .config import Config
from .stats import trend


def leave_one_year_out(yearly: pd.DataFrame, value: str = "value", year: str = "year") -> pd.DataFrame:
    """Theil-Sen/Mann-Kendall trend of `value` dropping each year in turn (plus the full series)."""
    rows = [{"dropped": None, **trend(yearly[year], yearly[value])}]
    for y in sorted(yearly[year].unique()):
        s = yearly[yearly[year] != y]
        rows.append({"dropped": int(y), **trend(s[year], s[value])})
    return pd.DataFrame(rows)


def h1_loyo(cfg: Config, timing: pd.DataFrame, proxies_yearly: pd.DataFrame | None = None) -> pd.DataFrame:
    parts = []
    for col in ["median", "peak", *[f"share_after_{c}" for c in cfg["geo"]["late_cutoffs"]]]:
        if col in timing:
            parts.append(leave_one_year_out(timing, col).assign(metric=f"geo_{col}"))
    if proxies_yearly is not None:
        for name, g in proxies_yearly.groupby("proxy"):
            parts.append(leave_one_year_out(g).assign(metric=name))
    out = pd.concat(parts, ignore_index=True)
    out.to_csv(cfg.table("robust_h1_loyo.csv"), index=False)
    return out


def h3_loyo(cfg: Config, t: pd.DataFrame) -> pd.DataFrame:
    years = sorted(t.loc[t["in_season"], "year"].unique())
    rows = []
    for drop in [None, *years]:
        keep = [y for y in years if y != drop]
        _, tr, _ = analysis.decoupling_regression(cfg, t, years=keep, tag=None)
        rows.append({"dropped": drop, **tr})
    out = pd.DataFrame(rows)
    out.to_csv(cfg.table("robust_h3_loyo.csv"), index=False)
    return out


def _variant(cfg: Config, **analysis_kw) -> Config:
    c = deepcopy(cfg)
    c.raw["analysis"].update(analysis_kw)
    return c


def h3_variants(cfg: Config, firms_daily=None, pm=None, met=None) -> pd.DataFrame:
    """Rebuild the table under alternative choices and refit the rising-beta test."""
    specs = {
        "baseline": {},
        "lags 0-1": {"fire_lags": [0, 1]},
        "lags 1-3": {"fire_lags": [1, 2, 3]},
        "no NW-wind weighting": {"weight_nw": False},
        "all polar sensors": {"fire_sensor": "ALL"},
        "Aqua only": {"fire_sensor": "Aqua"},
        "no UP/MP spillover control": {"spillover_control": False},
    }
    rows = []
    for label, kw in specs.items():
        c = _variant(cfg, **kw)
        t = analysis.build_table(c, firms_daily, pm, met, save=False)
        _, tr, _ = analysis.decoupling_regression(c, t, tag=None)
        rows.append({"variant": label, **tr})
        if label == "baseline":
            # Without 2020 (COVID) and without the Diwali window.
            _, tr2, _ = analysis.decoupling_regression(c, t.assign(in_season=t["in_season"] & (t["year"] != 2020)), tag=None)
            rows.append({"variant": "excluding 2020", **tr2})
            _, tr3, _ = analysis.decoupling_regression(c, t.assign(in_season=t["in_season"] & (t["diwali"] == 0)), tag=None)
            rows.append({"variant": "excluding Diwali +/-2 d", **tr3})
    out = pd.DataFrame(rows)
    out.to_csv(cfg.table("robust_h3_variants.csv"), index=False)
    return out


def drift_adjusted_trends(cfg: Config, proxies_yearly: pd.DataFrame, overpass: pd.DataFrame) -> pd.DataFrame:
    """Trend of each FIRMS proxy after partialling out that year's median overpass hour (orbit drift)."""
    med = overpass.pivot_table(index="year", columns="satellite", values="median")
    cov = {"aqua_terra_ratio": (med.get("Aqua") - med.get("Terra")) if {"Aqua", "Terra"} <= set(med.columns) else None,
           "aqua_snpp_ratio": None,          # the drift is the signal here, see aqua_dose_response
           "matched_late_only_share": None, "matched_persistence": None, "night_count_snpp": None}
    default = med["SNPP"] if "SNPP" in med else None
    rows = []
    for name, g in proxies_yearly.groupby("proxy"):
        g = g.set_index("year")["value"].dropna()
        x = cov.get(name, default)
        raw = trend(g.index, g.to_numpy())
        if x is None or len(g) < 5:
            rows.append({"proxy": name, "adjusted": False, **raw})
            continue
        x = x.reindex(g.index)
        ok = x.notna()
        if ok.sum() < 5 or x[ok].std() == 0:
            rows.append({"proxy": name, "adjusted": False, **raw})
            continue
        b = np.polyfit(x[ok], g[ok], 1)
        adj = g[ok] - np.polyval(b, x[ok]) + g[ok].mean()
        rows.append({"proxy": name, "adjusted": True, "drift_slope": float(b[0]), **trend(adj.index, adj.to_numpy())})
    out = pd.DataFrame(rows)
    out.to_csv(cfg.table("robust_drift_adjusted.csv"), index=False)
    return out


def aqua_dose_response(cfg: Config, yearly: pd.DataFrame, overpass: pd.DataFrame, baseline_to: int = 2022) -> pd.DataFrame:
    """Aqua/S-NPP ratio vs Aqua's median pass hour. Ratio relative to its 2016..baseline_to mean removes the
    MODIS-vs-VIIRS sensitivity gap; what is left is extra afternoon activity as Aqua samples later."""
    r = yearly[yearly["proxy"].eq("aqua_snpp_ratio")].set_index("year")[["value", "lo", "hi"]]
    hour = overpass[overpass["satellite"].eq("Aqua")].set_index("year")["median"].rename("aqua_hour")
    out = r.join(hour).dropna().reset_index()
    base = out.loc[out["year"] <= baseline_to, "value"].mean()
    out["relative"], out["relative_lo"], out["relative_hi"] = out["value"] / base, out["lo"] / base, out["hi"] / base
    out.to_csv(cfg.table("h1a_aqua_dose_response.csv"), index=False)
    return out
