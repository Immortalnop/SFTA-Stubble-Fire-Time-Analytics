"""H1-A: FIRMS-only timing proxies per season, with block-bootstrap CIs and trends.

Headline proxies (shown in the panel) vs appendix proxies (computed, not charted):
- night_count_snpp: absolute S-NPP night detections (~01:30). Fires lit late are still burning then; does not
  depend on how many daytime fires are seen.
- night_share_*: night / all detections of the same sensor. Also rises when daytime detections fall.
- low_frp_share_snpp: share of S-NPP daytime detections below the low-FRP threshold (just lit or dying).
- aqua_snpp_ratio: Aqua / S-NPP daytime detections. Aqua's overpass has drifted from ~13:50 to ~15:50 IST, so it
  samples later in the afternoon each year; the sensitivity gap MODIS vs VIIRS is the pre-2023 level.
- matched_*: S-NPP and NOAA-20 seen over the same ~25 km block and day; fires only the later pass sees vs only the
  earlier pass sees. Only informative about ignitions between the two passes (~50 min), not about the late shift.
- aqua_terra_ratio (appendix): Terra sees few Oct-Nov fires and both orbits drift, so the ratio is unstable.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config
from .stats import boot_ci, ratio, trend

PROXIES = {
    # name: (numerator columns, denominator columns, description)
    "night_share_snpp": (["snpp_N"], ["snpp_N", "snpp_D"], "S-NPP night detections / all S-NPP"),
    "night_share_modis": (["aqua_N", "terra_N"], ["aqua_N", "terra_N", "aqua_D", "terra_D"], "MODIS night / all MODIS"),
    "low_frp_share_snpp": (["snpp_D_low"], ["snpp_D"], "Share of S-NPP day detections below low-FRP threshold"),
    "aqua_snpp_ratio": (["aqua_D"], ["snpp_D"], "Aqua / S-NPP day detections (Aqua pass drifts later)"),
    "aqua_terra_ratio": (["aqua_D"], ["terra_D"], "Aqua day / Terra day detections (appendix)"),
}
COUNT_PROXY = ("night_count_snpp", "S-NPP night detections per season (count)")
MATCHED = {
    "matched_late_only_share": "Matched blocks: fires only the later pass sees / fires only one pass sees",
    "matched_persistence": "Matched blocks: share of earlier-pass fires still seen by the later pass",
}
HEADLINE = ["night_count_snpp", "night_share_snpp", "night_share_modis", "low_frp_share_snpp", "aqua_snpp_ratio",
            "matched_late_only_share", "matched_persistence"]
APPENDIX = ["aqua_terra_ratio"]
LABELS = {**{k: v[2] for k, v in PROXIES.items()}, COUNT_PROXY[0]: COUNT_PROXY[1], **MATCHED}


def season_days(cfg: Config, year: int) -> pd.DatetimeIndex:
    a, b = cfg.season_bounds(year, "season")
    return pd.date_range(a, b, freq="D")


def day_table(det: pd.DataFrame, days: pd.DatetimeIndex, low_frp: float) -> pd.DataFrame:
    """Region-level counts per day: <sat>_<D|N> and snpp_D_low."""
    cols = ["snpp_N", "snpp_D", "aqua_N", "terra_N", "aqua_D", "terra_D", "noaa20_D", "snpp_D_low"]
    d = det[det["date_ist"].isin(days)].copy()
    if d.empty:
        return pd.DataFrame(0.0, index=days, columns=cols)
    d["sat"] = d["satellite"].str.lower()
    counts = d.groupby(["date_ist", "sat", "daynight"]).size().unstack(["sat", "daynight"], fill_value=0)
    counts.columns = [f"{s}_{dn}" for s, dn in counts.columns]
    day = d[d["daynight"].eq("D")]
    low = day[day["sat"].eq("snpp") & (day["frp"] < low_frp)].groupby("date_ist").size().rename("snpp_D_low")
    out = pd.concat([counts, low], axis=1).reindex(days)
    for c in cols:
        out[c] = out[c].fillna(0) if c in out else 0.0
    return out


# ---------------------------------------------------------------- matched S-NPP / NOAA-20 blocks

def _cell(lat, lon, g):
    return np.floor(np.asarray(lat) / g).astype(np.int64) * 100000 + np.floor(np.asarray(lon) / g).astype(np.int64)


def matched_daily(det: pd.DataFrame, cell_deg: float = 0.01, block_deg: float = 0.25) -> pd.DataFrame:
    """Per IST date: cells seen only by the earlier pass, only by the later pass, or both.

    Only (date, block) pairs where both sensors detected something are used, as a stand-in for swath overlap
    (detections alone cannot show which areas a pass covered).
    """
    cols = ["early_only", "late_only", "both"]
    d = det[det["satellite"].isin(["SNPP", "NOAA20"]) & det["daynight"].eq("D")]
    if d.empty or d["satellite"].nunique() < 2:
        return pd.DataFrame(columns=cols)
    d = d.assign(cell=_cell(d["lat"], d["lon"], cell_deg), blk=_cell(d["lat"], d["lon"], block_deg))
    med = d.groupby(["date_ist", "blk", "satellite"])["hour_ist"].median().unstack("satellite").dropna()
    med = med.rename(columns={"SNPP": "m_s", "NOAA20": "m_n"}).reset_index()
    c = (d.groupby(["date_ist", "blk", "cell", "satellite"]).size().unstack("satellite", fill_value=0) > 0).reset_index()
    c = c.merge(med, on=["date_ist", "blk"])
    if c.empty:
        return pd.DataFrame(columns=cols)
    snpp_first = (c["m_s"] <= c["m_n"]).to_numpy()
    early = np.where(snpp_first, c["SNPP"], c["NOAA20"]).astype(bool)
    late = np.where(snpp_first, c["NOAA20"], c["SNPP"]).astype(bool)
    c["early_only"], c["late_only"], c["both"] = early & ~late, late & ~early, early & late
    return c.groupby("date_ist")[cols].sum()


def matched_yearly(cfg: Config, det: pd.DataFrame) -> pd.DataFrame:
    ac = cfg["analysis"]
    daily = matched_daily(det)
    rows = []
    if daily.empty:
        return pd.DataFrame(columns=["proxy", "year", "value", "lo", "hi", "n_days", "den_total"])
    for year, g in daily.groupby(daily.index.year):
        for name, num, den in (("matched_late_only_share", g["late_only"], g["early_only"] + g["late_only"]),
                               ("matched_persistence", g["both"], g["both"] + g["early_only"])):
            n, d_ = num.to_numpy(float), den.to_numpy(float)
            if d_.sum() == 0:
                continue
            est, lo, hi = boot_ci(ratio, [n, d_], ac["bootstrap_n"], ac["block_days"], ac["seed"] + int(year) + 3)
            rows.append({"proxy": name, "year": int(year), "value": est, "lo": lo, "hi": hi, "n_days": len(g), "den_total": d_.sum()})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- yearly table

def yearly_proxies(cfg: Config, det: pd.DataFrame | None = None) -> pd.DataFrame:
    if det is None:
        det = pd.read_parquet(cfg.processed("firms_detections.parquet"))
    fc, ac = cfg["firms"], cfg["analysis"]
    rows = []
    for year in cfg.years:
        tab = day_table(det, season_days(cfg, year), fc["low_frp_mw"])
        for name, (num_c, den_c, _) in PROXIES.items():
            num = tab[num_c].sum(axis=1).to_numpy(float)
            den = tab[den_c].sum(axis=1).to_numpy(float)
            if den.sum() == 0:
                est = lo = hi = np.nan
            else:
                est, lo, hi = boot_ci(ratio, [num, den], ac["bootstrap_n"], ac["block_days"], ac["seed"] + year)
            rows.append({"proxy": name, "year": year, "value": est, "lo": lo, "hi": hi, "n_days": len(tab), "den_total": den.sum()})
        night = tab["snpp_N"].to_numpy(float)
        est, lo, hi = boot_ci(lambda a: a.sum(), [night], ac["bootstrap_n"], ac["block_days"], ac["seed"] + year + 1)
        rows.append({"proxy": COUNT_PROXY[0], "year": year, "value": est, "lo": lo, "hi": hi, "n_days": len(tab), "den_total": night.sum()})
    out = pd.DataFrame(rows)
    if {"lat", "lon"} <= set(det.columns):
        out = pd.concat([out, matched_yearly(cfg, det[det["date_ist"].isin(np.concatenate([season_days(cfg, y) for y in cfg.years]))])], ignore_index=True)
    out.to_csv(cfg.table("h1a_proxies_yearly.csv"), index=False)
    return out


def proxy_trends(cfg: Config, yearly: pd.DataFrame) -> pd.DataFrame:
    rows = [{"proxy": p, "tier": "appendix" if p in APPENDIX else "headline", **trend(g["year"], g["value"])}
            for p, g in yearly.groupby("proxy")]
    out = pd.DataFrame(rows)
    out.to_csv(cfg.table("h1a_proxy_trends.csv"), index=False)
    return out


def overpass_times(cfg: Config, det: pd.DataFrame | None = None) -> pd.DataFrame:
    """Median IST time of daytime detections per satellite and season: tracks orbit drift."""
    if det is None:
        det = pd.read_parquet(cfg.processed("firms_detections.parquet"))
    day = det[det["daynight"].eq("D")]
    out = (day.groupby(["year", "satellite"])["hour_ist"]
              .agg(median="median", p10=lambda x: x.quantile(0.1), p90=lambda x: x.quantile(0.9), n="size")
              .reset_index())
    out.to_csv(cfg.table("overpass_times_ist.csv"), index=False)
    return out
