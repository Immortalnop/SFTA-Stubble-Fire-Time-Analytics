"""NASA FIRMS: download (area API), clean, and daily fire metrics.

Detections are tagged by the sensor we requested, not the CSV `satellite` column,
except MODIS where Terra and Aqua are split from that column.
"""
from __future__ import annotations

import io
import time
from datetime import date, timedelta

import numpy as np
import pandas as pd
import requests

from .config import Config, require_env
from .regions import assign_state, state_polygons

IST = pd.Timedelta(hours=5, minutes=30)
SOLAR = pd.Timedelta(hours=5)  # local mean solar time at ~75 deg E


# ---------------------------------------------------------------- download

def data_availability(cfg: Config, key: str) -> pd.DataFrame:
    url = f"{cfg['firms']['api']}/data_availability/csv/{key}/all"
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    df["min_date"] = pd.to_datetime(df["min_date"]).dt.date
    df["max_date"] = pd.to_datetime(df["max_date"]).dt.date
    return df.set_index("data_id")


def pick_source(day: date, sp: str, nrt: str, avail: pd.DataFrame) -> str | None:
    """Standard (SP) source where it covers the day, else NRT, else None."""
    for src in (sp, nrt):
        if src in avail.index and avail.at[src, "min_date"] <= day <= avail.at[src, "max_date"]:
            return src
    return None


def plan_requests(start: date, end: date, sp: str, nrt: str, avail: pd.DataFrame, day_range: int):
    """Split [start, end] into (source, first_day, n_days) runs of at most day_range days."""
    runs, cur = [], None
    d = start
    while d <= end:
        src = pick_source(d, sp, nrt, avail)
        if cur and cur[0] == src and cur[2] < day_range:
            cur[2] += 1
        else:
            if cur and cur[0]:
                runs.append(tuple(cur))
            cur = [src, d, 1]
        d += timedelta(days=1)
    if cur and cur[0]:
        runs.append(tuple(cur))
    return runs


def download(cfg: Config, years: list[int] | None = None, bbox=None, sources=None, subdir: str = "firms") -> None:
    key = require_env("FIRMS_MAP_KEY")
    fc = cfg["firms"]
    avail = data_availability(cfg, key)
    avail.to_csv(cfg.raw_dir(subdir) / "data_availability.csv")
    area = ",".join(str(v) for v in (bbox or cfg["region"]["bbox"]))
    for year in years or cfg.years:
        start, _ = cfg.season_bounds(year, "baseline")
        _, end = cfg.season_bounds(year, "season")
        end = min(end, date.today())
        for sensor, (sp, nrt) in (sources or fc["sources"]).items():
            out_dir = cfg.raw_dir(subdir, sensor)
            for src, first, n in plan_requests(start, end, sp, nrt, avail, fc["day_range"]):
                out = out_dir / f"{src}_{first.isoformat()}_{n}.csv"
                if out.exists():
                    continue
                url = f"{fc['api']}/area/csv/{key}/{src}/{area}/{n}/{first.isoformat()}"
                text = _get_with_retry(url)
                if not text.startswith("latitude"):
                    raise RuntimeError(f"FIRMS error for {out.name}: {text[:200]}")
                out.write_text(text)
                time.sleep(0.2)
            print(f"FIRMS {year} {sensor}: done")


def _get_with_retry(url: str, tries: int = 4) -> str:
    for i in range(tries):
        try:
            r = requests.get(url, timeout=120)
            if r.status_code == 429:
                time.sleep(60)
                continue
            r.raise_for_status()
            return r.text
        except requests.RequestException:
            if i == tries - 1:
                raise
            time.sleep(5 * (i + 1))
    raise RuntimeError(f"FIRMS request kept failing: {url}")


# ---------------------------------------------------------------- clean

def read_raw(cfg: Config, sensors=None, subdir: str = "firms") -> pd.DataFrame:
    frames = []
    for sensor in sensors or cfg["firms"]["sources"]:
        for f in sorted(cfg.raw_dir(subdir, sensor).glob("*.csv")):
            df = pd.read_csv(f, dtype={"confidence": str, "satellite": str, "version": str})
            if df.empty:
                continue
            df["sensor"] = sensor
            src, first, n = f.stem.rsplit("_", 2)
            df["source"] = src
            # Keep only the days this file was requested for (guards against API date semantics).
            first = date.fromisoformat(first)
            days = {(first + timedelta(days=i)).isoformat() for i in range(int(n))}
            frames.append(df[df["acq_date"].isin(days)])
    if not frames:
        raise SystemExit(f"No raw FIRMS files in raw/{subdir}. Run the matching download stage first.")
    return pd.concat(frames, ignore_index=True)


def normalise(df: pd.DataFrame) -> pd.DataFrame:
    """Uniform columns across MODIS/VIIRS and SP/NRT; time zones added."""
    t = df["acq_time"].astype(int)
    dt_utc = pd.to_datetime(df["acq_date"]) + pd.to_timedelta(t // 100, unit="h") + pd.to_timedelta(t % 100, unit="m")
    is_modis = df["sensor"].eq("MODIS")
    sat = df["sensor"].copy()
    sat[is_modis] = np.where(df.loc[is_modis, "satellite"].str.upper().str.startswith("A"), "Aqua", "Terra")
    conf = df["confidence"].astype(str).str.strip().str.lower()
    dt_ist = dt_utc + IST
    out = pd.DataFrame({
        "sensor": df["sensor"],
        "satellite": sat,
        "source": df["source"],
        "lat": df["latitude"].astype(float),
        "lon": df["longitude"].astype(float),
        "dt_utc": dt_utc,
        "dt_ist": dt_ist,
        "date_ist": dt_ist.dt.normalize(),
        "hour_ist": dt_ist.dt.hour + dt_ist.dt.minute / 60,
        "hour_solar": (dt_utc + SOLAR).dt.hour + (dt_utc + SOLAR).dt.minute / 60,
        "daynight": df["daynight"].astype(str).str.upper(),
        "frp": df["frp"].astype(float),
        "scan": df["scan"].astype(float),
        "conf_num": pd.to_numeric(conf.where(is_modis), errors="coerce"),
        "conf_cat": conf.where(~is_modis).str[:1],
        "type": pd.to_numeric(df["type"], errors="coerce") if "type" in df else np.nan,
        "is_sp": df["source"].str.endswith("_SP"),
    })
    return out


def _grid_key(lat, lon, g):
    return np.floor(np.asarray(lat) / g).astype(np.int64) * 100000 + np.floor(np.asarray(lon) / g).astype(np.int64)


def static_mask_keys(df: pd.DataFrame, g: float) -> set:
    """Grid cells flagged as static land sources (type 2) anywhere in the SP archive."""
    s = df[df["type"].eq(2)]
    return set(_grid_key(s["lat"], s["lon"], g))


def filter_detections(df: pd.DataFrame, fc: dict) -> pd.DataFrame:
    """Confidence, vegetation-fire and static-source filters, then swath dedup."""
    modis = df["sensor"].eq("MODIS")
    ok_conf = np.where(modis, df["conf_num"] >= fc["modis_min_confidence"], df["conf_cat"].isin(fc["viirs_confidence"]))
    df = df[ok_conf]
    static = static_mask_keys(df, fc["static_grid_deg"])
    # SP rows carry `type`; NRT rows do not, so fall back to the static-source mask.
    keep_sp = df["is_sp"] & df["type"].eq(0)
    keep_nrt = ~df["is_sp"] & ~pd.Series(_grid_key(df["lat"], df["lon"], fc["static_grid_deg"]), index=df.index).isin(static)
    df = df[keep_sp | keep_nrt].copy()
    # Overlapping swaths: same pixel, satellite, day and day/night pass -> keep the strongest.
    df["pix"] = _grid_key(df["lat"], df["lon"], fc["dedup_grid_deg"])
    df = df.sort_values("frp", ascending=False).drop_duplicates(["satellite", "date_ist", "daynight", "pix"])
    return df.sort_values("dt_utc").reset_index(drop=True)


def clean_raw(cfg: Config, raw: pd.DataFrame, bbox, states=None) -> pd.DataFrame:
    """Normalise, clip to bbox and state polygons, filter and dedup."""
    df = normalise(raw)
    w, s, e, n = bbox
    df = df[df["lon"].between(w, e) & df["lat"].between(s, n)]
    df = df.assign(state=assign_state(df["lon"].to_numpy(), df["lat"].to_numpy(), state_polygons(cfg, states)))
    df = df[df["state"].notna()]
    df = filter_detections(df, cfg["firms"])
    df["year"] = df["date_ist"].dt.year
    return df


def clean(cfg: Config) -> pd.DataFrame:
    df = clean_raw(cfg, read_raw(cfg), cfg["region"]["bbox"])
    df.to_parquet(cfg.processed("firms_detections.parquet"), index=False)
    print(f"FIRMS clean: {len(df):,} detections")
    return df


# ---------------------------------------------------------------- spillover (fires outside Punjab/Haryana)

def spill_download(cfg: Config, years: list[int] | None = None) -> None:
    """S-NPP detections over the spillover states' box (UP, MP): a covariate for H3, not part of H1."""
    sc = cfg["spillover"]
    download(cfg, years, bbox=sc["bbox"], sources={s: cfg["firms"]["sources"][s] for s in sc["sensors"]},
             subdir="firms_spill")


def spill_daily(cfg: Config) -> pd.DataFrame:
    """Daily vegetation-fire count per spillover state (IST dates)."""
    sc = cfg["spillover"]
    det = clean_raw(cfg, read_raw(cfg, sc["sensors"], "firms_spill"), sc["bbox"], sc["states"])
    daily = det.groupby(["date_ist", "state"]).size().rename("n").reset_index()
    daily.to_parquet(cfg.processed("firms_spill_daily.parquet"), index=False)
    print(f"Spillover fires: {len(det):,} detections, " + ", ".join(f"{k} {v:,}" for k, v in det["state"].value_counts().items()))
    return daily


# ---------------------------------------------------------------- metrics

def daily_metrics(cfg: Config, det: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per IST date x state x satellite x day/night."""
    if det is None:
        det = pd.read_parquet(cfg.processed("firms_detections.parquet"))
    g = det.groupby(["date_ist", "state", "satellite", "daynight"])
    daily = g.agg(
        n=("frp", "size"),
        pixels=("pix", "nunique"),
        frp_sum=("frp", "sum"),
        scan_median=("scan", "median"),
        hour_ist_median=("hour_ist", "median"),
        n_low_frp=("frp", lambda x: int((x < cfg["firms"]["low_frp_mw"]).sum())),
    ).reset_index()
    daily["year"] = daily["date_ist"].dt.year
    daily.to_parquet(cfg.processed("firms_daily.parquet"), index=False)
    print(f"FIRMS daily: {len(daily):,} rows")
    return daily
