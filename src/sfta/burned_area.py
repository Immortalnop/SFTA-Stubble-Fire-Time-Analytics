"""Sentinel-2 dNBR burned area on cropland (Earth Engine) and FIRMS reconciliation (H2 estimator 3).

Not run against Earth Engine yet: needs `earthengine authenticate`, EE_PROJECT in .env, and a labels CSV.
Reconciliation is pixel-level: burned pixels within `viirs_buffer_m` of any VIIRS detection count as detected.
"""
from __future__ import annotations

import os
from datetime import date

import numpy as np
import pandas as pd

from .config import Config

VIIRS = ["SNPP", "NOAA20", "NOAA21"]


def init_ee(cfg: Config):
    import ee

    ee.Initialize(project=os.environ.get(cfg["burned_area"]["ee_project_env"]) or None)
    return ee


def best_threshold(dnbr: np.ndarray, burned: np.ndarray) -> tuple[float, float]:
    """dNBR cut-off maximising Youden's J (sensitivity + specificity - 1); returns (threshold, J)."""
    dnbr, burned = np.asarray(dnbr, float), np.asarray(burned, bool)
    ok = np.isfinite(dnbr)
    dnbr, burned = dnbr[ok], burned[ok]
    if burned.all() or (~burned).all():
        raise ValueError("Labels need both burned and unburned fields")
    best = (np.nan, -1.0)
    for t in np.unique(dnbr):
        pred = dnbr >= t
        j = pred[burned].mean() + (~pred[~burned]).mean() - 1
        if j > best[1]:
            best = (float(t), float(j))
    return best


def _season_window(year: int, md: list[str]) -> tuple[str, str]:
    return f"{year}-{md[0]}", f"{year}-{md[1]}"


def nbr_composite(ee, geom, start: str, end: str):
    def prep(img):
        scl = img.select("SCL")
        ok = scl.neq(3).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10)).And(scl.neq(11))  # shadow, cloud, cirrus, snow
        return img.updateMask(ok).normalizedDifference(["B8", "B12"]).rename("nbr")

    col = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(geom).filterDate(start, end).map(prep))
    return col.median(), col.count().rename("n_obs")


def dnbr_image(ee, cfg: Config, year: int, geom):
    ba = cfg["burned_area"]
    pre, _ = nbr_composite(ee, geom, *_season_window(year, ba["pre"]))
    post, _ = nbr_composite(ee, geom, *_season_window(year, ba["post"]))
    crop = ee.Image(ba["worldcover"]).select("Map").eq(ba["cropland_class"])
    return pre.subtract(post).rename("dnbr").updateMask(crop)


def district_geometry(ee, names: list[str]):
    fc = (ee.FeatureCollection("FAO/GAUL/2015/level2")
          .filter(ee.Filter.inList("ADM1_NAME", ["Punjab", "Haryana"]))
          .filter(ee.Filter.inList("ADM2_NAME", names)))
    n = fc.size().getInfo()
    if n != len(names):
        got = fc.aggregate_array("ADM2_NAME").getInfo()
        raise SystemExit(f"GAUL matched {n} of {len(names)} districts: {got}. Fix burned_area.districts in config.yaml.")
    return fc


def calibrate(cfg: Config, ee=None) -> float:
    """Sample dNBR at the hand-labelled fields and pick the threshold."""
    ee = ee or init_ee(cfg)
    ba = cfg["burned_area"]
    lab = pd.read_csv(ba["labels_csv"])
    vals = []
    for year, g in lab.groupby("year"):
        pts = ee.FeatureCollection([ee.Feature(ee.Geometry.Point([r.lon, r.lat]), {"burned": int(r.burned)}) for r in g.itertuples()])
        geom = pts.geometry().bounds()
        s = dnbr_image(ee, cfg, int(year), geom).sampleRegions(pts, ["burned"], scale=ba["scale_m"])
        vals.append(pd.DataFrame([f["properties"] for f in s.getInfo()["features"]]))
    v = pd.concat(vals)
    thr, j = best_threshold(v["dnbr"], v["burned"])
    pd.DataFrame([{"threshold": thr, "youden_j": j, "n_labels": len(v)}]).to_csv(cfg.table("dnbr_calibration.csv"), index=False)
    print(f"dNBR threshold {thr:.3f} (J={j:.2f}, n={len(v)})")
    return thr


def viirs_points(ee, det: pd.DataFrame, year: int):
    d = det[(det["year"] == year) & det["satellite"].isin(VIIRS)]
    return ee.FeatureCollection([ee.Feature(ee.Geometry.Point([float(r.lon), float(r.lat)])) for r in d.itertuples()])


def run(cfg: Config, threshold: float | None = None) -> pd.DataFrame:
    ee = init_ee(cfg)
    ba = cfg["burned_area"]
    thr = threshold if threshold is not None else calibrate(cfg, ee)
    det = pd.read_parquet(cfg.processed("firms_detections.parquet"))
    districts = district_geometry(ee, ba["districts"])
    rows = []
    for year in cfg.years:
        if date(year, 12, 10) > date.today() and year == cfg.years[-1]:
            continue  # post-season window not complete yet
        pts = viirs_points(ee, det, year)
        detected = pts.map(lambda f: f.buffer(ba["viirs_buffer_m"])).geometry()
        for name in ba["districts"]:
            geom = districts.filter(ee.Filter.eq("ADM2_NAME", name)).geometry()
            burned = dnbr_image(ee, cfg, year, geom).gte(thr)
            area = ee.Image.pixelArea().divide(1e4)
            red = dict(reducer=ee.Reducer.sum(), geometry=geom, scale=ba["scale_m"], maxPixels=1e10, tileScale=4)
            total = area.updateMask(burned).reduceRegion(**red).get("area")
            tagged = area.updateMask(burned.clip(detected)).reduceRegion(**red).get("area")
            n = pts.filterBounds(geom).size()
            t, g, k = ee.List([total, tagged, n]).getInfo()
            rows.append({"year": year, "district": name, "total_ha": t or 0.0, "tagged_ha": g or 0.0, "n_viirs": k,
                         "low_confidence": year in ba["low_confidence_years"]})
            print(f"burned area {year} {name}: {t or 0:.0f} ha, {g or 0:.0f} ha near VIIRS")
    out = pd.DataFrame(rows)
    out["missed_share_upper"] = 1 - out["tagged_ha"] / out["total_ha"].replace(0, np.nan)
    out.to_csv(cfg.table("burned_area_by_year.csv"), index=False)
    return out
