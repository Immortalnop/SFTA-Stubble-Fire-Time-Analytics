"""ERA5 hourly meteorology over Delhi (Copernicus CDS) -> daily controls in IST."""
from __future__ import annotations

import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import xarray as xr

from .config import Config, require_env

IST = pd.Timedelta(hours=5, minutes=30)

SINGLE_VARS = [
    "boundary_layer_height", "10m_u_component_of_wind", "10m_v_component_of_wind",
    "2m_temperature", "2m_dewpoint_temperature", "total_precipitation",
]


def _request(cfg: Config, year: int | list[int], pressure: bool) -> dict:
    ec = cfg["era5"]
    years = [year] if isinstance(year, int) else list(year)
    req = {
        "product_type": ["reanalysis"],
        "year": [str(y) for y in years],
        "month": ec["months"],
        "day": [f"{d:02d}" for d in range(1, 32)],
        "time": [f"{h:02d}:00" for h in range(0, 24, ec.get("time_step_hours", 1))],
        "area": ec["area"],
        "data_format": "netcdf",
        "download_format": "unarchived",
    }
    if pressure:
        req |= {"variable": ["u_component_of_wind", "v_component_of_wind"], "pressure_level": ["850"]}
    else:
        req |= {"variable": SINGLE_VARS}
    return req


# era5_alt holds files fetched by side scripts running next to era5-queue; both folders are read everywhere.
ERA5_DIRS = ("era5", "era5_alt")


def _files(cfg: Config, pattern: str) -> list[Path]:
    return [f for d in ERA5_DIRS for f in sorted(cfg.raw_dir(d).glob(pattern))]


def file_years(f: Path) -> list[int]:
    """Years held by sfc_2019.nc or bulk_sfc_2017_2020.nc."""
    parts = f.stem.split("_")
    if parts[0] == "bulk":
        y0, y1 = int(parts[-2]), int(parts[-1])
        return list(range(y0, y1 + 1))
    return [int(parts[-1])]


def bulk_covers(cfg: Config, kind: str, year: int) -> bool:
    """True if a combined file bulk_<kind>_<y0>_<y1>.nc already holds `year`."""
    return any(year in file_years(f) for f in _files(cfg, f"bulk_{kind}_*_*.nc"))


def have_year(cfg: Config, kind: str, year: int) -> bool:
    return any(f.name == f"{kind}_{year}.nc" for f in _files(cfg, f"{kind}_*.nc")) or bulk_covers(cfg, kind, year)


def chunk_years(years: list[int], size: int) -> list[list[int]]:
    ys = sorted(years)
    return [ys[i:i + size] for i in range(0, len(ys), size)]


COLLECTION = {"sfc": ("reanalysis-era5-single-levels", False), "pl850": ("reanalysis-era5-pressure-levels", True)}


def consecutive_runs(years: list[int]) -> list[list[int]]:
    runs: list[list[int]] = []
    for y in sorted(years):
        if runs and y == runs[-1][-1] + 1:
            runs[-1].append(y)
        else:
            runs.append([y])
    return runs


def max_years(cfg: Config, client, pressure: bool, probe_year: int) -> int:
    """Years that fit in one request under the CDS cost limit, from the server's own estimate (nothing is queued)."""
    est = client.estimate_costs(COLLECTION["pl850" if pressure else "sfc"][0], _request(cfg, probe_year, pressure))
    return max(1, int(est["limit"] // est["cost"]))


def download_queue(cfg: Config, years: list[int] | None = None) -> None:
    """One request at a time per dataset (CDS allows one queued request per dataset), the two datasets side by side.

    First picks up jobs already on the server (queued, running, or finished but never downloaded), so their waiting
    time is not lost, then fetches the missing years in multi-year chunks sized by the server's cost limit.
    """
    import cdsapi
    import ecmwf.datastores as dstore
    from concurrent.futures import ThreadPoolExecutor

    years = years or cfg.years
    out_dir = cfg.raw_dir("era5")
    url, key = require_env("CDSAPI_URL"), require_env("CDSAPI_KEY")

    def have(kind, y):
        return have_year(cfg, kind, y)

    def fetch(kind, ys, legacy, depth=0):
        coll, pressure = COLLECTION[kind]
        out = out_dir / f"bulk_{kind}_{ys[0]}_{ys[-1]}.nc"
        tmp = out.with_suffix(".part")
        while True:
            try:
                legacy.retrieve(coll, _request(cfg, ys, pressure)).download(str(tmp))
                break
            except requests.HTTPError as e:
                msg = str(e)
                if "cost limits exceeded" in msg and len(ys) > 1:
                    print(f"ERA5 {kind} {ys[0]}-{ys[-1]}: too large, splitting", flush=True)
                    half = len(ys) // 2
                    fetch(kind, ys[:half], legacy, depth + 1)
                    fetch(kind, ys[half:], legacy, depth + 1)
                    return
                if "queued requests" in msg:      # another job of ours still holds the dataset's slot
                    time.sleep(120)
                    continue
                raise
        tmp.rename(out)
        print(f"ERA5 {out.name}: done", flush=True)

    def worker(kind):
        coll, pressure = COLLECTION[kind]
        client = dstore.Client(url=url, key=key)
        legacy = cdsapi.Client(url=url, key=key)
        for status in ("successful", "running", "accepted"):
            for rid in client.get_jobs(status=status, limit=50).request_ids:
                r = client.get_remote(rid)
                ys = sorted(int(y) for y in (r.request or {}).get("year", []))
                if r.collection_id != coll or not ys or all(have(kind, y) for y in ys):
                    continue
                if ys != list(range(ys[0], ys[-1] + 1)):      # file names hold a year range only
                    continue
                name = f"{kind}_{ys[0]}.nc" if len(ys) == 1 else f"bulk_{kind}_{ys[0]}_{ys[-1]}.nc"
                out = out_dir / name
                print(f"ERA5 adopting {status} job {rid[:8]} ({kind} {ys[0]}-{ys[-1]})", flush=True)
                try:
                    r.download(str(out.with_suffix(".part")))
                except Exception as e:      # finished results expire on the server after a while
                    print(f"ERA5 job {rid[:8]}: could not download ({str(e)[:120]})", flush=True)
                    continue
                out.with_suffix(".part").rename(out)
                print(f"ERA5 {out.name}: done", flush=True)
        todo = [y for y in years if not have(kind, y)]
        if not todo:
            return
        per = max_years(cfg, client, pressure, todo[0])
        for run in consecutive_runs(todo):
            for chunk in chunk_years(run, per):
                fetch(kind, chunk, legacy)

    with ThreadPoolExecutor(2) as ex:
        for f in [ex.submit(worker, k) for k in COLLECTION]:
            f.result()


def status(cfg: Config) -> pd.DataFrame:
    """Print the account's CDS jobs and which years are on disk per dataset."""
    import ecmwf.datastores as dstore

    client = dstore.Client(url=require_env("CDSAPI_URL"), key=require_env("CDSAPI_KEY"))
    kinds = {coll: kind for kind, (coll, _) in COLLECTION.items()}
    print("CDS jobs:")
    for st in ("accepted", "running", "successful", "failed"):
        for rid in client.get_jobs(status=st, limit=20).request_ids:
            r = client.get_remote(rid)
            ys = (r.request or {}).get("year", [])
            if r.collection_id in kinds:
                print(f"  {st:<10} {rid[:8]}  {kinds[r.collection_id]:<5}  {ys[0]}-{ys[-1]}" if ys else f"  {st} {rid[:8]}")
    cover = pd.DataFrame({kind: [have_year(cfg, kind, y) for y in cfg.years] for kind in COLLECTION}, index=cfg.years)
    print("On disk (raw/era5, raw/era5_alt):")
    for kind in COLLECTION:
        missing = [y for y in cfg.years if not cover.at[y, kind]]
        print(f"  {kind:<5} missing: {missing or 'none'}")
    return cover


def download(cfg: Config, years: list[int] | None = None) -> None:
    import cdsapi

    client = cdsapi.Client(url=require_env("CDSAPI_URL"), key=require_env("CDSAPI_KEY"))
    out_dir = cfg.raw_dir("era5")
    for year in years or cfg.years:
        for pressure, ds in ((False, "reanalysis-era5-single-levels"), (True, "reanalysis-era5-pressure-levels")):
            kind = "pl850" if pressure else "sfc"
            out = out_dir / f"{kind}_{year}.nc"
            if have_year(cfg, kind, year):
                continue
            tmp = out.with_suffix(".part")
            client.retrieve(ds, _request(cfg, year, pressure)).download(str(tmp))
            tmp.rename(out)
            print(f"ERA5 {out.name}: done")


def _open(path: Path) -> xr.Dataset:
    """Open a CDS download; mixed instant/accumulated variables arrive as a zip of netCDFs."""
    if zipfile.is_zipfile(path):
        ex = path.with_suffix("")
        with zipfile.ZipFile(path) as z:
            z.extractall(ex)
        return xr.merge([xr.open_dataset(p, engine="h5netcdf") for p in sorted(ex.glob("*.nc"))], compat="override")
    return xr.open_dataset(path, engine="h5netcdf")


def _hourly_frame(ds: xr.Dataset) -> pd.DataFrame:
    if "expver" in ds.dims:        # ERA5 / ERA5T overlap: take whichever version has data
        ds = ds.mean("expver", skipna=True)
    tname = "valid_time" if "valid_time" in ds.coords else "time"
    dims = [d for d in ("latitude", "longitude", "pressure_level") if d in ds.dims]
    df = ds.mean(dim=dims).to_dataframe()
    df.index = pd.to_datetime(df.index.get_level_values(tname) if isinstance(df.index, pd.MultiIndex) else df.index)
    return df.drop(columns=[c for c in ("number", "expver", "pressure_level") if c in df.columns])


def relative_humidity(t_c, td_c):
    """Magnus formula, percent."""
    return 100 * np.exp(17.625 * td_c / (243.04 + td_c)) / np.exp(17.625 * t_c / (243.04 + t_c))


def wind_from_deg(u, v):
    """Meteorological direction the wind blows FROM, degrees."""
    return (270 - np.degrees(np.arctan2(v, u))) % 360


def daily_from_hourly(sfc: pd.DataFrame, pl: pd.DataFrame, nw_sector) -> pd.DataFrame:
    h = sfc.join(pl[["u", "v"]].rename(columns={"u": "u850", "v": "v850"}), how="left")
    h.index = h.index + IST
    t = h["t2m"] - 273.15
    td = h["d2m"] - 273.15
    wdir = wind_from_deg(h["u850"], h["v850"])
    a, b = nw_sector
    hh = pd.DataFrame({
        "blh": h["blh"],
        "ws10": np.hypot(h["u10"], h["v10"]),
        "t2m": t,
        "rh": relative_humidity(t, td),
        "tp": h["tp"] * 1000,
        "u850": h["u850"],
        "v850": h["v850"],
        "nw": ((wdir >= a) & (wdir <= b)).astype(float).where(wdir.notna()),   # NaN, not 0, without 850 hPa wind
    }, index=h.index)
    d = hh.resample("D").agg({"blh": "mean", "ws10": "mean", "t2m": "mean", "rh": "mean", "tp": "mean",
                              "u850": "mean", "v850": "mean", "nw": "mean"})
    d["tp"] = d["tp"] * 24                # mean hourly accumulation x 24 h (works for hourly and 3-hourly data)
    d = d.rename(columns={"nw": "nw_frac"})
    d["blh_min"] = hh["blh"].resample("D").min()
    d = d.dropna(subset=["blh"])          # resample fills the gaps between requested seasons
    d.index.name = "date"
    return d.reset_index()


def _hourly_all(cfg: Config, kind: str) -> pd.DataFrame:
    files = _files(cfg, f"{kind}_*.nc") + _files(cfg, f"bulk_{kind}_*.nc")
    if not files:
        return pd.DataFrame()
    df = pd.concat([_hourly_frame(_open(f)) for f in files])
    return df[~df.index.duplicated()].sort_index()


def process(cfg: Config) -> pd.DataFrame:
    sfc, pl = _hourly_all(cfg, "sfc"), _hourly_all(cfg, "pl850")
    if sfc.empty or pl.empty:
        raise SystemExit("No ERA5 files. Run: python -m sfta era5-download (or era5-bulk)")
    out = daily_from_hourly(sfc, pl, cfg["era5"]["nw_sector_deg"])
    years = sorted(set(out["date"].dt.year))
    missing = [y for y in cfg.years if y not in years]
    if missing:
        print(f"ERA5: no data yet for {missing}")
    out.to_parquet(cfg.processed("era5_delhi_daily.parquet"), index=False)
    print(f"ERA5 daily: {len(out):,} days, years {years[0]}-{years[-1]}")
    return out
