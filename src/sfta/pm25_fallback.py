"""Fill PM2.5 gaps from CSV exports (CPCB CCR portal downloads, Princeton CPCB 2015-2019, ...).

File layouts were not verified, so the column mapping is explicit. Each CSV needs a station column, a timestamp
column and a PM2.5 column (hourly or finer; sub-hourly rows are averaged to hours). Timestamps are taken as IST
unless `--utc` is given. Output goes where `pm25.read_raw` already looks, so `pm25-process` picks it up.
"""
from __future__ import annotations

import zlib
from pathlib import Path

import pandas as pd

from .config import Config

IST = pd.Timedelta(hours=5, minutes=30)


def station_id(name: str) -> int:
    """Stable negative id per station name, distinct from OpenAQ's positive sensor ids."""
    return -int(zlib.crc32(name.strip().lower().encode()) % 10**9) - 1


def load_csv_dir(cfg: Config, folder: str | Path, station: str, time: str, value: str, utc: bool = False,
                 only: list[str] | None = None) -> int:
    """Convert CSVs under `folder` to per-year/sensor parquet files in the OpenAQ raw layout; returns rows kept."""
    kept = 0
    frames = []
    for f in sorted(Path(folder).rglob("*.csv")):
        df = pd.read_csv(f, usecols=[station, time, value])
        df.columns = ["station", "time", "value"]
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        frames.append(df.dropna())
    if not frames:
        raise SystemExit(f"No CSV files under {folder}")
    df = pd.concat(frames, ignore_index=True)
    if only:
        df = df[df["station"].isin(only)]
    t = pd.to_datetime(df["time"], errors="coerce")
    df = df.assign(dt=t).dropna(subset=["dt"])
    if t.dt.tz is not None:
        df["dt"] = df["dt"].dt.tz_convert("UTC").dt.tz_localize(None) + IST   # -> IST wall clock
    elif utc:
        df["dt"] = df["dt"] + IST
    # Hours are IST hours (they start at :30 UTC), matching how pm25.read_raw reads OpenAQ periods.
    df["hour_utc"] = df["dt"].dt.floor("h") - IST
    hourly = df.groupby(["station", "hour_utc"])["value"].mean().reset_index()
    for (station_name, year), g in hourly.groupby(["station", (hourly["hour_utc"] + IST).dt.year]):
        out = cfg.raw_dir("openaq", str(int(year))) / f"sensor_{station_id(station_name)}.parquet"
        new = pd.DataFrame({"dt_utc": g["hour_utc"].dt.strftime("%Y-%m-%dT%H:%M:%SZ"), "value": g["value"]})
        if out.exists():
            new = pd.concat([pd.read_parquet(out), new]).drop_duplicates("dt_utc")
        new.to_parquet(out, index=False)
        kept += len(g)
    return kept
