"""Smoke tracers over Punjab/Haryana (TROPOMI CO, MAIAC AOD) via Earth Engine, and the H3 tracer substitution.

Not run against Earth Engine yet (see burned_area.py for setup).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import analysis
from .burned_area import init_ee
from .config import Config


def _daily_mean(ee, collection: str, band: str, geom, start: str, end: str, scale: int) -> pd.DataFrame:
    col = ee.ImageCollection(collection).select(band).filterDate(start, end).filterBounds(geom)

    def per_image(img):
        v = img.reduceRegion(ee.Reducer.mean(), geom, scale, maxPixels=1e9).get(band)
        return ee.Feature(None, {"t": img.date().format("YYYY-MM-dd"), "v": v})

    feats = col.map(per_image).filter(ee.Filter.notNull(["v"])).getInfo()["features"]
    df = pd.DataFrame([f["properties"] for f in feats])
    if df.empty:
        return pd.DataFrame(columns=["date", "v"])
    df["date"] = pd.to_datetime(df["t"])
    return df.groupby("date")["v"].mean().reset_index()


def download(cfg: Config) -> pd.DataFrame:
    ee = init_ee(cfg)
    tc = cfg["tracers"]
    w, s, e, n = cfg["region"]["bbox"]
    geom = ee.Geometry.Rectangle([w, s, e, n])
    frames = []
    for year in range(tc["start_year"], cfg.years[-1] + 1):
        a = cfg.season_bounds(year, "baseline")[0].isoformat()
        b = cfg.season_bounds(year, "season")[1].isoformat()
        co = _daily_mean(ee, tc["co_collection"], tc["co_band"], geom, a, b, 5000).rename(columns={"v": "tropomi_co"})
        aod = _daily_mean(ee, tc["aod_collection"], tc["aod_band"], geom, a, b, 5000).rename(columns={"v": "maiac_aod"})
        frames.append(co.merge(aod, on="date", how="outer"))
        print(f"tracers {year}: {len(frames[-1])} days")
    out = pd.concat(frames, ignore_index=True)
    out.to_parquet(cfg.processed("tracers_daily.parquet"), index=False)
    return out


def tracer_table(cfg: Config, t: pd.DataFrame, tracers: pd.DataFrame) -> pd.DataFrame:
    t = t.merge(tracers, on="date", how="left")
    lags = cfg["analysis"]["fire_lags"]
    for col in ("tropomi_co", "maiac_aod"):
        t[f"log_{col}"] = np.log(t[col].where(t[col] > 0))
        t[f"{col}_dl"] = pd.concat([t.groupby("year")[f"log_{col}"].shift(k) for k in lags], axis=1).mean(axis=1, skipna=False)
    return t


def substitution(cfg: Config, t: pd.DataFrame, tracers: pd.DataFrame | None = None) -> pd.DataFrame:
    """Rising-beta test with fires replaced by tracers, and with evening/daytime PM2.5 as outcome."""
    tracers = tracers if tracers is not None else pd.read_parquet(cfg.processed("tracers_daily.parquet"))
    t = tracer_table(cfg, t, tracers)
    runs = [("fires | PM2.5 24h", "log_pm", "fire_dl"),
            ("fires | PM2.5 evening/night", "log_pm_night", "fire_dl"),
            ("fires | PM2.5 daytime", "log_pm_day", "fire_dl"),
            ("TROPOMI CO | PM2.5 24h", "log_pm", "tropomi_co_dl"),
            ("MAIAC AOD | PM2.5 24h", "log_pm", "maiac_aod_dl")]
    rows = []
    for label, dep, fire in runs:
        if dep not in t or fire not in t or t[fire].notna().sum() < 100:
            continue
        beta, trend, _ = analysis.decoupling_regression(cfg, t, dep=dep, fire=fire, tag=None)
        rows.append({"test": label, **trend, "mean_beta": beta["beta"].mean()})
    out = pd.DataFrame(rows)
    out.to_csv(cfg.table("h3_tracer_substitution.csv"), index=False)
    return out
