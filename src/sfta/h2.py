"""H2: share of fires lit after the last afternoon polar overpass (estimators 1 and 2; 3 is burned_area)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config
from .stats import block_indices

SENSOR_SETS = {"snpp_only": ["SNPP"], "all_polar": ["SNPP", "NOAA20", "NOAA21", "Aqua", "Terra"]}


# ---------------------------------------------------------------- overpass times from FIRMS

def daily_top(cfg: Config, det: pd.DataFrame, sensors: list[str] | None = None) -> pd.DataFrame:
    """T_op per IST date: latest afternoon daytime acquisition (q99 of hour per satellite, max over satellites)."""
    h2 = cfg["h2"]
    lo, hi = h2["afternoon_hours"]
    d = det[det["daynight"].eq("D") & det["hour_ist"].between(lo, hi)]
    if sensors:
        d = d[d["satellite"].isin(sensors)]
    per_sat = d.groupby(["date_ist", "satellite"])["hour_ist"].quantile(h2["top_quantile"]).reset_index()
    top = per_sat.groupby("date_ist")["hour_ist"].max().rename("t_op").reset_index()
    top["year"] = top["date_ist"].dt.year
    return top


def fill_top(top: pd.DataFrame, dates: pd.Series) -> pd.Series:
    """T_op for each date; days with no afternoon detection get that year's median."""
    s = top.set_index("date_ist")["t_op"]
    yr_med = top.groupby("year")["t_op"].median()
    got = dates.map(s)
    fallback = dates.dt.year.map(yr_med)
    return got.fillna(fallback)


def pass_times(det: pd.DataFrame, sensors: list[str]) -> pd.DataFrame:
    """Median IST hour per year x satellite x day/night pass."""
    d = det[det["satellite"].isin(sensors)]
    return d.groupby(["year", "satellite", "daynight"])["hour_ist"].median().rename("hour").reset_index()


# ---------------------------------------------------------------- estimator 1

def estimator1(cfg: Config, ev: pd.DataFrame, top: pd.DataFrame) -> pd.DataFrame:
    ac = cfg["analysis"]
    ev = ev.assign(t_op=fill_top(top, ev["date_ist"]))
    ev["after"] = ev["hour_ist"] > ev["t_op"]
    rows = []
    for year, y in ev.groupby("year"):
        days = [x for _, x in y.groupby("date_ist")]
        rng = np.random.default_rng(ac["seed"] + int(year) + 7)
        draws = [np.concatenate([days[i]["after"].to_numpy() for i in block_indices(len(days), ac["block_days"], rng)]).mean()
                 for _ in range(min(ac["bootstrap_n"], 500))]
        rows.append({"year": int(year), "est1_share": y["after"].mean(), "est1_lo": np.percentile(draws, 2.5),
                     "est1_hi": np.percentile(draws, 97.5), "n_events": len(y), "t_op_median": y["t_op"].median()})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- estimator 2

def missed_probability(start_h: np.ndarray, dur_h: np.ndarray, passes: np.ndarray, rng, slot_jitter: float, pass_jitter: float) -> float:
    """Share of fires with no polar pass inside [ignition, ignition + duration] (passes repeat daily: +/-24 h)."""
    if len(start_h) == 0 or len(passes) == 0:
        return np.nan
    s = start_h + rng.uniform(-slot_jitter, slot_jitter, len(start_h))
    p = np.concatenate([passes, passes + 24])[None, :] + rng.normal(0, pass_jitter, (len(start_h), 2 * len(passes)))
    seen = ((p >= s[:, None]) & (p <= (s + dur_h)[:, None])).any(axis=1)
    return float(1 - seen.mean())


def estimator2(cfg: Config, ev: pd.DataFrame, det: pd.DataFrame) -> pd.DataFrame:
    h2, ac = cfg["h2"], cfg["analysis"]
    rows = []
    for set_name, sensors in SENSOR_SETS.items():
        pt = pass_times(det, sensors)
        for year, y in ev.groupby("year"):
            passes = pt.loc[pt["year"].eq(year), "hour"].to_numpy()
            days = [x for _, x in y.groupby("date_ist")]
            for mult in h2["duration_multipliers"]:
                rng = np.random.default_rng(ac["seed"] + int(year) * 13 + int(mult * 10))
                draws = []
                for _ in range(h2["mc_draws"]):
                    pick = pd.concat([days[i] for i in block_indices(len(days), ac["block_days"], rng)])
                    draws.append(missed_probability(pick["hour_ist"].to_numpy(), pick["duration_h"].to_numpy() * mult,
                                                    passes, rng, h2["slot_jitter_h"], h2["pass_jitter_h"]))
                draws = np.array(draws, float)
                rows.append({"sensor_set": set_name, "year": int(year), "duration_mult": mult,
                             "est2_missed": np.nanmean(draws) if np.isfinite(draws).any() else np.nan,
                             "est2_lo": np.nanpercentile(draws, 2.5) if np.isfinite(draws).any() else np.nan,
                             "est2_hi": np.nanpercentile(draws, 97.5) if np.isfinite(draws).any() else np.nan,
                             "n_passes": len(passes)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- table

def run(cfg: Config, ev: pd.DataFrame | None = None, det: pd.DataFrame | None = None) -> pd.DataFrame:
    ev = ev if ev is not None else pd.read_parquet(cfg.processed("geo_events.parquet"))
    det = det if det is not None else pd.read_parquet(cfg.processed("firms_detections.parquet"))
    top = daily_top(cfg, det, SENSOR_SETS["all_polar"])
    e1 = estimator1(cfg, ev, top)
    e2 = estimator2(cfg, ev, det)
    base = e2[(e2["sensor_set"] == "all_polar") & (e2["duration_mult"] == 1.0)].drop(columns=["sensor_set", "duration_mult"])
    table = e1.merge(base, on="year", how="outer")
    try:
        ba = pd.read_csv(cfg.table("burned_area_by_year.csv"))
        agg = ba.groupby("year")[["total_ha", "tagged_ha"]].sum()
        agg["est3_missed_upper"] = 1 - agg["tagged_ha"] / agg["total_ha"]
        table = table.merge(agg[["est3_missed_upper"]].reset_index(), on="year", how="left")
    except FileNotFoundError:
        table["est3_missed_upper"] = np.nan
    table.to_csv(cfg.table("h2_missed_share.csv"), index=False)
    e2.to_csv(cfg.table("h2_estimator2_sensitivity.csv"), index=False)
    top.to_csv(cfg.table("t_op_daily.csv"), index=False)
    return table
