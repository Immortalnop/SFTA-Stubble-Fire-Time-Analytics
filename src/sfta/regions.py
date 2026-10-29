"""Punjab + Haryana boundary from GADM 4.1, and point-in-region tests."""
from __future__ import annotations

import io
import json
import zipfile

import numpy as np
import requests
import shapely
from shapely.geometry import shape
from shapely.ops import unary_union

from .config import Config


def _gadm_path(cfg: Config):
    return cfg.raw_dir("boundaries") / "gadm41_IND_1.json"


def download_boundaries(cfg: Config) -> None:
    out = _gadm_path(cfg)
    if out.exists():
        return
    r = requests.get(cfg["region"]["gadm_url"], timeout=300)
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        name = next(n for n in z.namelist() if n.endswith(".json"))
        out.write_bytes(z.read(name))


def state_polygons(cfg: Config, states: list[str] | None = None) -> dict[str, shapely.Geometry]:
    """Return {state name: geometry} for the configured states (or the given GADM names)."""
    download_boundaries(cfg)
    gj = json.loads(_gadm_path(cfg).read_text(encoding="utf-8"))
    wanted = set(states or cfg["region"]["states"])
    polys: dict[str, list] = {}
    for feat in gj["features"]:
        name = feat["properties"]["NAME_1"]
        if name in wanted:
            polys.setdefault(name, []).append(shape(feat["geometry"]))
    missing = wanted - polys.keys()
    if missing:
        raise ValueError(f"States not found in GADM: {missing}")
    return {k: unary_union(v) for k, v in polys.items()}


def assign_state(lon: np.ndarray, lat: np.ndarray, polys: dict[str, shapely.Geometry]) -> np.ndarray:
    """Name of the containing state per point, or None outside all states."""
    out = np.full(len(lon), None, dtype=object)
    for name, geom in polys.items():
        shapely.prepare(geom)
        out[shapely.contains_xy(geom, lon, lat)] = name
    return out


def delhi_polygon(cfg: Config) -> shapely.Geometry:
    """NCT of Delhi from the same GADM file (state name contains 'Delhi')."""
    download_boundaries(cfg)
    gj = json.loads(_gadm_path(cfg).read_text(encoding="utf-8"))
    geoms = [shape(f["geometry"]) for f in gj["features"] if "delhi" in f["properties"]["NAME_1"].lower()]
    if not geoms:
        raise ValueError("Delhi not found in GADM level 1")
    return unary_union(geoms)
