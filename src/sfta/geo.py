"""Geostationary fire timing: LSA SAF MSG-IODC FRP-PIXEL download, parse, event clustering, H1-B stats.

Files are `HDF5_LSASAF_MSG-IODC_FRP-PIXEL-ListProduct_IODC-Disk_<YYYYMMDDHHMM>` (UTC slot start, 15 min).
Dataset names inside the HDF5 were not documented anywhere reachable, so the reader matches names
case-insensitively and fails with the list of names it did see. Run `python -m sfta geo-inspect` first.
"""
from __future__ import annotations

import re
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta

import h5py
import numpy as np
import pandas as pd
import requests

from .config import Config, require_env
from .regions import assign_state, state_polygons
from .stats import block_indices, trend

IST = pd.Timedelta(hours=5, minutes=30)
LIST_KIND = "FRP-PIXEL-ListProduct"
NAME_ALIASES = {"lat": ("latitude", "lat"), "lon": ("longitude", "lon"), "frp": ("frp", "frp_pixel", "fire_radiative_power")}


# ---------------------------------------------------------------- download

def _session() -> requests.Session:
    s = requests.Session()
    s.auth = (require_env("LSASAF_USER"), require_env("LSASAF_PASSWORD"))
    return s


def list_day(s: requests.Session, base: str, d: date) -> list[str]:
    """URLs of the ListProduct files for one UTC day (empty if the folder is missing)."""
    url = f"{base}/{d:%Y/%m/%d}/"
    r = s.get(url, timeout=60)
    if r.status_code == 404:
        return []
    r.raise_for_status()
    root = re.match(r"https://[^/]+", base).group(0)
    return sorted({root + h for h in re.findall(r'href="(/PRODUCTS[^"]*)"', r.text) if LIST_KIND in h})


def slot_of(name: str) -> pd.Timestamp:
    return pd.to_datetime(re.search(r"_(\d{12})(?:\.\w+)?$", name).group(1), format="%Y%m%d%H%M")


def _fetch(s: requests.Session, url: str, out) -> None:
    for i in range(4):
        try:
            r = s.get(url, timeout=120)
            r.raise_for_status()
            body = r.content
            if body[:3] == b"BZh":
                import bz2
                body = bz2.decompress(body)
            out.write_bytes(body)
            return
        except requests.RequestException:
            if i == 3:
                raise
            time.sleep(3 * (i + 1))


def download(cfg: Config, years: list[int] | None = None) -> None:
    g = cfg["geo"]
    s = _session()
    lo, hi = g["years"]
    for year in years or cfg.years:
        if not lo <= year <= hi:
            continue
        a, b = cfg.season_bounds(year)
        # IST season days cover UTC from a-1 18:30; take one extra UTC day each side.
        d, end, jobs = a - timedelta(days=1), b + timedelta(days=1), []
        out_dir = cfg.raw_dir("geo", str(year))
        while d <= end:
            for url in list_day(s, g["base_url"], d):
                out = out_dir / url.rsplit("/", 1)[1]
                if not out.exists():
                    jobs.append((url, out))
            d += timedelta(days=1)
        with ThreadPoolExecutor(g["threads"]) as ex:
            list(ex.map(lambda j: _fetch(s, *j), jobs))
        print(f"GEO {year}: {len(jobs)} new files")


# ---------------------------------------------------------------- parse

def _datasets(f: h5py.File) -> dict[str, str]:
    found: dict[str, str] = {}

    def visit(name, obj):  # must return None: visititems stops at the first non-None value
        if isinstance(obj, h5py.Dataset):
            found.setdefault(name.split("/")[-1].lower(), name)

    f.visititems(visit)
    return found


def _physical(ds: h5py.Dataset) -> np.ndarray:
    """Raw array scaled by SCALING_FACTOR/OFFSET attributes when present (convention checked by caller)."""
    raw = ds[...].astype("float64").ravel()
    sf = float(np.ravel(ds.attrs.get("SCALING_FACTOR", 1))[0]) or 1.0
    off = float(np.ravel(ds.attrs.get("OFFSET", 0))[0])
    miss = ds.attrs.get("MISSING_VALUE")
    if miss is not None:
        raw[raw == float(np.ravel(miss)[0])] = np.nan
    return raw, sf, off


def inspect_file(path) -> str:
    lines = [str(path)]
    with h5py.File(path) as f:
        for name, full in _datasets(f).items():
            ds = f[full]
            lines.append(f"  {full}  shape={ds.shape} dtype={ds.dtype} attrs={ {k: (v if np.size(v) < 4 else '...') for k, v in ds.attrs.items()} }")
        lines.append(f"  file attrs: { {k: (v if np.size(v) < 4 else '...') for k, v in f.attrs.items()} }")
    return "\n".join(lines)


def read_slot(path) -> pd.DataFrame:
    """Fire pixels of one slot: lat, lon, frp (+ any other equally long 1-D numeric dataset)."""
    with h5py.File(path) as f:
        names = _datasets(f)
        cols = {}
        for key, aliases in NAME_ALIASES.items():
            hit = next((names[a] for a in aliases if a in names), None)
            if hit is None:
                raise KeyError(f"{path}: no dataset for '{key}' among {sorted(names)}")
            cols[key] = hit
        arrs = {}
        for key, full in cols.items():
            raw, sf, off = _physical(f[full])
            arrs[key] = (raw, sf, off)
        n = len(arrs["lat"][0])
        if n == 0:
            return pd.DataFrame(columns=["lat", "lon", "frp"])
        # Scaling convention check: coordinates must land on Earth either as raw/sf or raw*sf.
        def apply(key, mode):
            raw, sf, off = arrs[key]
            return (raw / sf if mode == "div" else raw * sf) + off
        for mode in ("div", "mul"):
            lat, lon = apply("lat", mode), apply("lon", mode)
            if np.nanmax(np.abs(lat)) <= 90 and np.nanmax(np.abs(lon)) <= 180:
                break
        else:
            raise ValueError(f"{path}: coordinates out of range under both scaling conventions; run geo-inspect")
        df = pd.DataFrame({"lat": lat, "lon": lon, "frp": apply("frp", mode)})
        for name, full in names.items():
            if full in cols.values():
                continue
            ds = f[full]
            if ds.ndim == 1 and ds.shape[0] == n and ds.dtype.kind in "iuf":
                df[name] = ds[...]
    return df


def parse(cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    """All downloaded slots -> (pixel detections in Punjab/Haryana, slot table incl. empty slots)."""
    polys = state_polygons(cfg)
    w, s, e, n = cfg["region"]["bbox"]
    pixels, slots = [], []
    for f in sorted(cfg.raw_dir("geo").glob("*/*")):
        slot = slot_of(f.name)
        slots.append(slot)
        df = read_slot(f)
        if df.empty:
            continue
        df = df[df["lon"].between(w, e) & df["lat"].between(s, n) & (df["frp"] > 0)]
        if df.empty:
            continue
        df = df.assign(slot_utc=slot, state=assign_state(df["lon"].to_numpy(), df["lat"].to_numpy(), polys))
        pixels.append(df[df["state"].notna()])
    if not slots:
        raise SystemExit("No raw GEO files. Run: python -m sfta geo-download")
    pix = pd.concat(pixels, ignore_index=True) if pixels else pd.DataFrame(columns=["lat", "lon", "frp", "slot_utc", "state"])
    slot_tab = pd.DataFrame({"slot_utc": sorted(slots)})
    slot_tab["slot_ist"] = slot_tab["slot_utc"] + IST
    pix.to_parquet(cfg.processed("geo_pixels.parquet"), index=False)
    slot_tab.to_parquet(cfg.processed("geo_slots.parquet"), index=False)
    print(f"GEO parse: {len(slot_tab):,} slots, {len(pix):,} fire pixels")
    return pix, slot_tab


# ---------------------------------------------------------------- events

class _UF:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, i):
        while self.p[i] != i:
            self.p[i] = self.p[self.p[i]]
            i = self.p[i]
        return i

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def cluster_events(pix: pd.DataFrame, cell_deg: float, max_gap_slots: int, slot_minutes: int = 15) -> pd.DataFrame:
    """Link detections in the same or 8-neighbour cells within max_gap_slots; one row per event."""
    if pix.empty:
        return pd.DataFrame(columns=["event", "first_utc", "last_utc", "duration_h", "peak_frp", "n_pixels", "n_slots", "lat", "lon", "state"])
    d = pix.sort_values("slot_utc").reset_index(drop=True)
    slot_i = ((d["slot_utc"] - d["slot_utc"].min()) / pd.Timedelta(minutes=slot_minutes)).round().astype(int).to_numpy()
    ci = np.floor(d["lat"].to_numpy() / cell_deg).astype(int)
    cj = np.floor(d["lon"].to_numpy() / cell_deg).astype(int)
    uf = _UF(len(d))
    recent: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for r in range(len(d)):
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                for other, s in recent.get((ci[r] + di, cj[r] + dj), ()):
                    if slot_i[r] - s <= max_gap_slots:
                        uf.union(other, r)
        lst = recent.setdefault((ci[r], cj[r]), [])
        lst.append((r, slot_i[r]))
        if len(lst) > 64:  # keep only entries that can still link
            recent[(ci[r], cj[r])] = [x for x in lst if slot_i[r] - x[1] <= max_gap_slots]
    d["event"] = [uf.find(i) for i in range(len(d))]
    g = d.groupby("event")
    ev = g.agg(first_utc=("slot_utc", "min"), last_utc=("slot_utc", "max"), peak_frp=("frp", "max"),
               n_pixels=("frp", "size"), lat=("lat", "mean"), lon=("lon", "mean"), state=("state", "first")).reset_index()
    ev["n_slots"] = g["slot_utc"].nunique().to_numpy()
    ev["duration_h"] = ((ev["last_utc"] - ev["first_utc"]) / pd.Timedelta(hours=1)) + slot_minutes / 60
    return ev


def build_events(cfg: Config, pix: pd.DataFrame | None = None) -> pd.DataFrame:
    g = cfg["geo"]
    if pix is None:
        pix = pd.read_parquet(cfg.processed("geo_pixels.parquet"))
    ev = cluster_events(pix, g["cell_deg"], g["max_gap_slots"], g["slot_minutes"])
    # Ignition proxy: middle of the first detecting slot, in IST.
    ev["first_ist"] = ev["first_utc"] + IST + pd.Timedelta(minutes=g["slot_minutes"] / 2)
    ev["date_ist"] = ev["first_ist"].dt.normalize()
    ev["hour_ist"] = ev["first_ist"].dt.hour + ev["first_ist"].dt.minute / 60
    ev["year"] = ev["date_ist"].dt.year
    md = ev["date_ist"].dt.strftime("%m-%d")
    ev = ev[(md >= cfg["season"]["start"]) & (md <= cfg["season"]["end"])]
    ev.to_parquet(cfg.processed("geo_events.parquet"), index=False)
    print(f"GEO events: {len(ev):,} in season windows")
    return ev


# ---------------------------------------------------------------- H1-B statistics

def observed_slots_per_hour(slots: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Slots with a file per year and IST hour: the denominator that stops missing data reading as no fire."""
    s = slots.copy()
    md = s["slot_ist"].dt.strftime("%m-%d")
    s = s[(md >= cfg["season"]["start"]) & (md <= cfg["season"]["end"])]
    s["year"], s["hour"] = s["slot_ist"].dt.year, s["slot_ist"].dt.hour
    return s.groupby(["year", "hour"]).size().rename("slots").reset_index()


def event_weights(ev: pd.DataFrame, obs: pd.DataFrame) -> np.ndarray:
    """1 / relative observation effort in the event's IST hour, so gaps in the archive do not shift the density."""
    o = obs.set_index(["year", "hour"])["slots"]
    rel = o / o.groupby(level="year").transform("mean")
    key = pd.MultiIndex.from_arrays([ev["year"], ev["hour_ist"].astype(int)])
    w = 1.0 / rel.reindex(key).to_numpy()
    return np.where(np.isfinite(w), w, 1.0)


def _stats_from(hours: np.ndarray, w: np.ndarray, cuts) -> dict:
    from scipy.stats import gaussian_kde

    out = {"n": len(hours)}
    if len(hours) < 5:
        return {**out, "median": np.nan, "peak": np.nan, **{f"share_after_{c}": np.nan for c in cuts}}
    order = np.argsort(hours)
    cw = np.cumsum(w[order]) / w.sum()
    out["median"] = float(hours[order][np.searchsorted(cw, 0.5)])
    grid = np.linspace(0, 24, 481)
    out["peak"] = float(grid[np.argmax(gaussian_kde(hours, weights=w, bw_method=0.25)(grid))])
    for c in cuts:
        out[f"share_after_{c}"] = float(w[hours > c].sum() / w.sum())
    return out


def yearly_timing(cfg: Config, ev: pd.DataFrame, slots: pd.DataFrame) -> pd.DataFrame:
    """Per year: median, KDE peak hour, share after cutoffs; day-block bootstrap CIs."""
    g, ac = cfg["geo"], cfg["analysis"]
    cuts = g["late_cutoffs"]
    ev = ev.assign(w=event_weights(ev, observed_slots_per_hour(slots, cfg)))
    rows = []
    for year, y in ev.groupby("year"):
        base = _stats_from(y["hour_ist"].to_numpy(), y["w"].to_numpy(), cuts)
        days = [x for _, x in y.groupby("date_ist")]
        rng = np.random.default_rng(ac["seed"] + int(year))
        draws = []
        for _ in range(min(ac["bootstrap_n"], 500)):
            pick = pd.concat([days[i] for i in block_indices(len(days), ac["block_days"], rng)])
            draws.append(_stats_from(pick["hour_ist"].to_numpy(), pick["w"].to_numpy(), cuts))
        row = {"year": int(year), **base}
        for k in base:
            if k == "n":
                continue
            v = np.array([d[k] for d in draws], float)
            row[f"{k}_lo"], row[f"{k}_hi"] = (np.nanpercentile(v, 2.5), np.nanpercentile(v, 97.5)) if np.isfinite(v).any() else (np.nan, np.nan)
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(cfg.table("h1b_timing_yearly.csv"), index=False)
    return out


def timing_trends(cfg: Config, yearly: pd.DataFrame) -> pd.DataFrame:
    cols = ["median", "peak"] + [f"share_after_{c}" for c in cfg["geo"]["late_cutoffs"]]
    out = pd.DataFrame([{"metric": c, **trend(yearly["year"], yearly[c])} for c in cols])
    out.to_csv(cfg.table("h1b_timing_trends.csv"), index=False)
    return out
