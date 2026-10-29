"""Delhi PM2.5: OpenAQ v3 download (CPCB stations), QC, and daily Delhi series."""
from __future__ import annotations

import re
import time

import numpy as np
import pandas as pd
import requests

from .config import Config, require_env

IST = pd.Timedelta(hours=5, minutes=30)


# ---------------------------------------------------------------- download

class OpenAQ:
    def __init__(self, base: str, key: str):
        self.base = base
        self.s = requests.Session()
        self.s.headers["X-API-Key"] = key

    def get(self, path: str, **params) -> dict:
        for attempt in range(6):
            try:
                r = self.s.get(f"{self.base}{path}", params=params, timeout=120)
            except (requests.ConnectionError, requests.Timeout):
                if attempt == 5:
                    raise
                time.sleep(10 * (attempt + 1))
                continue
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep(int(r.headers.get("x-ratelimit-reset", 60)) if r.status_code == 429 else 10 * (attempt + 1))
                continue
            r.raise_for_status()
            # Stay under 60 requests/minute.
            if int(r.headers.get("x-ratelimit-remaining", 10)) < 3:
                time.sleep(int(r.headers.get("x-ratelimit-reset", 60)))
            else:
                time.sleep(1.05)
            return r.json()
        raise RuntimeError(f"OpenAQ kept failing: {path} {params}")


def find_sensors(cfg: Config, api: OpenAQ) -> pd.DataFrame:
    pc = cfg["pm25"]
    w, s, e, n = cfg["delhi"]["bbox"]
    res = api.get("/locations", bbox=f"{w},{s},{e},{n}", parameters_id=pc["parameter_id"], limit=1000)["results"]
    pat = re.compile(pc["provider_pattern"], re.I)
    rows = []
    for loc in res:
        provider = (loc.get("provider") or {}).get("name", "")
        for sen in loc.get("sensors", []):
            if sen["parameter"]["id"] == pc["parameter_id"]:
                rows.append({
                    "location_id": loc["id"], "location": loc["name"], "provider": provider,
                    "sensor_id": sen["id"], "lat": loc["coordinates"]["latitude"], "lon": loc["coordinates"]["longitude"],
                    "first": (loc.get("datetimeFirst") or {}).get("utc"), "last": (loc.get("datetimeLast") or {}).get("utc"),
                    "provider_match": bool(pat.search(provider)),
                })
    return pd.DataFrame(rows)


def download(cfg: Config, years: list[int] | None = None) -> None:
    api = OpenAQ(cfg["pm25"]["openaq_api"], require_env("OPENAQ_API_KEY"))
    sensors = find_sensors(cfg, api)
    sensors.to_csv(cfg.raw_dir("openaq") / "sensors.csv", index=False)
    use = sensors[sensors["provider_match"]]
    print(f"OpenAQ: {len(sensors)} PM2.5 sensors in Delhi box, {len(use)} match provider pattern")
    for year in years or cfg.years:
        start, _ = cfg.season_bounds(year, "baseline")
        _, end = cfg.season_bounds(year, "season")
        t0 = (pd.Timestamp(start) - IST).strftime("%Y-%m-%dT%H:%M:%SZ")
        t1 = (pd.Timestamp(end) + pd.Timedelta(days=2) - IST).strftime("%Y-%m-%dT%H:%M:%SZ")
        for sid in [*use["sensor_id"], *cfg["pm25"]["reference_sensors"]]:
            out = cfg.raw_dir("openaq", str(year)) / f"sensor_{sid}.parquet"
            if out.exists():
                continue
            rows, page = [], 1
            while True:
                res = api.get(f"/sensors/{sid}/hours", datetime_from=t0, datetime_to=t1, limit=1000, page=page)["results"]
                rows += [{"dt_utc": r["period"]["datetimeFrom"]["utc"], "value": r["value"]} for r in res]
                if len(res) < 1000:
                    break
                page += 1
            pd.DataFrame(rows, columns=["dt_utc", "value"]).to_parquet(out, index=False)
        print(f"OpenAQ {year}: done")


# ---------------------------------------------------------------- QC + daily

def station_map(cfg: Config) -> dict[int, int]:
    """OpenAQ sensor id -> station id, for provider-matched sensors inside Delhi.

    Sensors within ~100 m of each other (same site reported by CPCB, DPCC and caaqm) count as one station,
    so a site is not weighted twice in the Delhi median. NCR towns in the search box (Gurugram, Noida, ...) drop out.
    """
    from .regions import delhi_polygon
    import shapely

    s = pd.read_csv(cfg.raw_dir("openaq") / "sensors.csv")
    s = s[s["provider_match"]]
    s = s[shapely.contains_xy(delhi_polygon(cfg), s["lon"].to_numpy(), s["lat"].to_numpy())]
    key = s["lat"].round(3).astype(str) + "," + s["lon"].round(3).astype(str)
    return dict(zip(s["sensor_id"].astype(int), pd.factorize(key)[0] + 1))


def read_raw(cfg: Config) -> pd.DataFrame:
    """Hourly values per station (sensor ids merged by site; non-OpenAQ negative ids kept as their own station)."""
    smap = station_map(cfg) if (cfg.raw_dir("openaq") / "sensors.csv").exists() else {}
    frames = []
    for f in cfg.raw_dir("openaq").glob("*/sensor_*.parquet"):
        sid = int(f.stem.split("_")[1])
        station = smap.get(sid) if smap and sid > 0 else sid
        if station is None:
            continue
        df = pd.read_parquet(f)
        if len(df):
            frames.append(df.assign(sensor_id=station))
    if not frames:
        raise SystemExit("No raw OpenAQ files. Run: python -m sfta pm25-download")
    df = pd.concat(frames, ignore_index=True)
    df["dt_ist"] = pd.to_datetime(df["dt_utc"], utc=True).dt.tz_localize(None) + IST
    df = df.drop(columns="dt_utc").groupby(["sensor_id", "dt_ist"], as_index=False)["value"].mean()
    return df


def flat_run_mask(values: pd.Series, min_run: int) -> pd.Series:
    """True where a value belongs to a run of >= min_run identical consecutive values."""
    grp = (values != values.shift()).cumsum()
    size = values.groupby(grp).transform("size")
    return size >= min_run


def qc_hourly(df: pd.DataFrame, pc: dict) -> pd.DataFrame:
    lo, hi = pc["valid_range"]
    df = df[(df["value"] > lo) & (df["value"] <= hi)].sort_values(["sensor_id", "dt_ist"])
    flat = df.groupby("sensor_id", group_keys=False)["value"].apply(lambda v: flat_run_mask(v, pc["flat_run_hours"]))
    return df[~flat.reindex(df.index).fillna(False).astype(bool)]


def delhi_daily(df: pd.DataFrame, pc: dict) -> pd.DataFrame:
    """Median across stations of station daily means (>=75% hours), plus an evening/night mean."""
    df = df.copy()
    df["date"] = df["dt_ist"].dt.normalize()
    day = df.groupby(["sensor_id", "date"])["value"].agg(["mean", "size"])
    day = day[day["size"] >= pc["min_daily_coverage"] * 24]

    eve, morn = pc["night_hours"]
    h = df["dt_ist"].dt.hour
    night = df[(h >= eve) | (h < morn)].copy()
    night["date"] = (night["dt_ist"] - pd.Timedelta(hours=eve)).dt.normalize()
    n_hours = 24 - eve + morn
    nt = night.groupby(["sensor_id", "date"])["value"].agg(["mean", "size"])
    nt = nt[nt["size"] >= pc["min_daily_coverage"] * n_hours]

    dh = df[(h >= morn) & (h < eve)]
    dy = dh.groupby(["sensor_id", "date"])["value"].agg(["mean", "size"])
    dy = dy[dy["size"] >= pc["min_daily_coverage"] * (eve - morn)]

    out = pd.DataFrame({
        "pm25": day.groupby("date")["mean"].median(),
        "n_stations": day.groupby("date")["mean"].size(),
        "pm25_night": nt.groupby("date")["mean"].median(),
        "pm25_day": dy.groupby("date")["mean"].median(),
    })
    out.index.name = "date"
    return out.reset_index()


def process_reference(cfg: Config) -> pd.DataFrame:
    """Single-station reference series (US Embassy, New Delhi) kept apart from the CPCB multi-station median.

    OpenAQ holds no CPCB data from ~Nov 2022 to Dec 2024, but this station is continuous 2016-2026. The two
    sensor ids (State Air until 2016-11-09, AirNow after) are one site, so they are merged into one station.
    """
    frames = []
    for sid in cfg["pm25"]["reference_sensors"]:
        for f in cfg.raw_dir("openaq").glob(f"*/sensor_{sid}.parquet"):
            df = pd.read_parquet(f)
            if len(df):
                frames.append(df)
    if not frames:
        raise SystemExit("No reference-station files. Run: python -m sfta pm25-download")
    df = pd.concat(frames, ignore_index=True)
    df["dt_ist"] = pd.to_datetime(df["dt_utc"], utc=True).dt.tz_localize(None) + IST
    df = df.drop(columns="dt_utc").groupby("dt_ist", as_index=False)["value"].mean().assign(sensor_id=1)
    daily = delhi_daily(qc_hourly(df, cfg["pm25"]), cfg["pm25"])
    daily.to_parquet(cfg.processed("pm25_ref_daily.parquet"), index=False)
    print(f"PM2.5 reference station daily: {len(daily):,} days")
    return daily


def process(cfg: Config) -> pd.DataFrame:
    hourly = qc_hourly(read_raw(cfg), cfg["pm25"])
    daily = delhi_daily(hourly, cfg["pm25"])
    daily.to_parquet(cfg.processed("pm25_delhi_daily.parquet"), index=False)
    print(f"PM2.5 daily: {len(daily):,} days, median {daily['n_stations'].median():.0f} stations/day")
    return daily
