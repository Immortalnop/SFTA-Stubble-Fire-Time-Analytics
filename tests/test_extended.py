import h5py
import numpy as np
import pandas as pd
import pytest

from sfta import analysis, burned_area, geo, h2, pm25_fallback, robustness
from sfta.config import load_config


@pytest.fixture
def cfg(tmp_path):
    c = load_config()
    c.data_dir = tmp_path / "data"
    c.outputs = tmp_path / "out"
    c.raw["analysis"]["bootstrap_n"] = 100
    c.raw["h2"]["mc_draws"] = 100
    return c


# ---------------------------------------------------------------- GEO

def _pix(rows):
    return pd.DataFrame(rows, columns=["slot_utc", "lat", "lon", "frp"]).assign(state="Punjab", slot_utc=lambda d: pd.to_datetime(d["slot_utc"]))


def test_cluster_links_neighbours_and_gaps_but_not_far_or_late():
    pix = _pix([
        ("2024-10-20 09:00", 30.000, 75.000, 10.0),
        ("2024-10-20 09:15", 30.040, 75.000, 20.0),   # adjacent cell, next slot: same event
        ("2024-10-20 09:45", 30.040, 75.040, 15.0),   # 2-slot gap, adjacent cell: same event
        ("2024-10-20 09:00", 31.000, 76.000, 5.0),    # far away: new event
        ("2024-10-20 12:00", 30.000, 75.000, 8.0),    # same cell but 12 slots later: new event
    ])
    ev = geo.cluster_events(pix, cell_deg=0.04, max_gap_slots=3)
    assert len(ev) == 3
    big = ev.sort_values("n_pixels").iloc[-1]
    assert big["n_pixels"] == 3 and big["peak_frp"] == 20.0
    assert big["first_utc"] == pd.Timestamp("2024-10-20 09:00")
    assert big["duration_h"] == pytest.approx(1.0)       # 09:00 .. 09:45 + one 15-min slot


def test_read_slot_applies_scaling_and_checks_range(tmp_path):
    p = tmp_path / "HDF5_LSASAF_MSG-IODC_FRP-PIXEL-ListProduct_IODC-Disk_202410201200"
    with h5py.File(p, "w") as f:
        for name, vals in {"LATITUDE": [30.5, 29.9], "LONGITUDE": [75.25, 76.0], "FRP": [12.5, 3.0]}.items():
            ds = f.create_dataset(name, data=(np.array(vals) * 100).astype("i4"))
            ds.attrs["SCALING_FACTOR"] = 100.0
        f.create_dataset("EXTRA", data=np.array([1, 2], dtype="i2"))
    df = geo.read_slot(p)
    assert df["lat"].tolist() == pytest.approx([30.5, 29.9])
    assert df["frp"].tolist() == pytest.approx([12.5, 3.0])
    assert "extra" in df
    assert geo.slot_of(p.name) == pd.Timestamp("2024-10-20 12:00")


def test_read_slot_names_missing_dataset(tmp_path):
    p = tmp_path / "x_202410201200"
    with h5py.File(p, "w") as f:
        f.create_dataset("LATITUDE", data=[1.0])
    with pytest.raises(KeyError, match="lon"):
        geo.read_slot(p)


def _events(late_by_year):
    rng = np.random.default_rng(0)
    rows = []
    for year, centre in late_by_year.items():
        for d in pd.date_range(f"{year}-10-10", f"{year}-11-10"):
            for h in rng.normal(centre, 1.0, 12):
                rows.append({"year": year, "date_ist": d, "hour_ist": float(np.clip(h, 6, 23)), "duration_h": 1.0})
    return pd.DataFrame(rows)


def _slots(years):
    s = [t for y in years for t in pd.date_range(f"{y}-09-15", f"{y}-11-30 23:45", freq="15min")]
    return pd.DataFrame({"slot_utc": s, "slot_ist": pd.DatetimeIndex(s) + geo.IST})


def test_yearly_timing_detects_later_shift(cfg):
    years = [2019, 2020, 2021, 2022, 2023, 2024]
    ev = _events({y: 13.5 + 0.7 * (y - 2019) for y in years})
    yr = geo.yearly_timing(cfg, ev, _slots(years))
    assert yr["median"].iloc[-1] - yr["median"].iloc[0] == pytest.approx(3.5, abs=0.3)
    assert yr["share_after_17"].iloc[-1] > yr["share_after_17"].iloc[0]
    tr = geo.timing_trends(cfg, yr).set_index("metric")
    assert tr.loc["median", "slope"] == pytest.approx(0.7, abs=0.1)


# ---------------------------------------------------------------- H2

def test_missed_probability_known_cases():
    rng = np.random.default_rng(0)
    start = np.full(500, 16.0)
    assert h2.missed_probability(start, np.full(500, 0.5), np.array([13.5]), rng, 0.0, 0.0) == 1.0   # pass at 13:30 never overlaps
    assert h2.missed_probability(start, np.full(500, 3.0), np.array([17.0]), rng, 0.0, 0.0) == 0.0   # pass at 17:00 inside window
    assert h2.missed_probability(np.full(500, 23.0), np.full(500, 3.0), np.array([1.5]), rng, 0.0, 0.0) == 0.0  # next-day pass


def test_estimator1_share_after_top(cfg):
    top = pd.DataFrame({"date_ist": pd.date_range("2024-10-01", periods=10), "t_op": 14.0, "year": 2024})
    ev = pd.DataFrame({"year": 2024, "date_ist": np.repeat(pd.date_range("2024-10-01", periods=10), 10),
                       "hour_ist": np.tile(np.r_[np.full(7, 12.0), np.full(3, 16.0)], 10)})
    e1 = h2.estimator1(cfg, ev, top)
    assert e1["est1_share"].iloc[0] == pytest.approx(0.3)


def test_daily_top_uses_latest_afternoon_pass(cfg):
    det = pd.DataFrame({"date_ist": pd.Timestamp("2024-10-20"), "satellite": ["SNPP"] * 3 + ["Aqua"] * 3, "daynight": "D",
                        "hour_ist": [13.2, 13.4, 13.5, 14.6, 14.7, 14.8]})
    top = h2.daily_top(cfg, det)
    assert top["t_op"].iloc[0] == pytest.approx(14.8, abs=0.05)


# ---------------------------------------------------------------- burned area

def test_best_threshold_separates_classes():
    rng = np.random.default_rng(1)
    d = np.r_[rng.normal(0.05, 0.03, 200), rng.normal(0.35, 0.05, 200)]
    y = np.r_[np.zeros(200), np.ones(200)].astype(bool)
    thr, j = burned_area.best_threshold(d, y)
    assert 0.12 < thr < 0.3 and j > 0.9
    with pytest.raises(ValueError):
        burned_area.best_threshold(d, np.ones(400, bool))


# ---------------------------------------------------------------- robustness

def test_leave_one_year_out_flags_influential_year():
    y = pd.DataFrame({"year": range(2016, 2026), "value": [0.1] * 9 + [0.9]})
    out = robustness.leave_one_year_out(y).set_index("dropped")
    assert out.loc[2025, "slope"] == pytest.approx(0.0, abs=1e-9)
    assert out.loc[2020, "slope"] >= 0


def test_h3_loyo_and_variants_run(cfg):
    rng = np.random.default_rng(0)
    cfg.raw["years"] = [2018, 2024]
    rows, mets = [], []
    for y in range(2018, 2025):
        days = pd.date_range(f"{y}-09-01", f"{y}-11-30")
        n = rng.gamma(2.0, 200, len(days)) * np.linspace(0.2, 2, len(days))
        rows.append(pd.DataFrame({"date_ist": days, "satellite": "SNPP", "n": n.round(), "frp_sum": n * 5, "daynight": "D", "state": "Punjab"}))
        mets.append(pd.DataFrame({"date": days, "blh": rng.normal(800, 100, len(days)), "ws10": rng.normal(2, .3, len(days)),
                                  "t2m": rng.normal(25, 2, len(days)), "rh": rng.normal(60, 5, len(days)), "tp": rng.exponential(1, len(days)),
                                  "u850": 1.0, "v850": -1.0, "nw_frac": 0.8, "blh_min": 200.0}))
    fd, met = pd.concat(rows), pd.concat(mets)
    pm = pd.DataFrame({"date": met["date"], "pm25": np.exp(4 + rng.normal(0, .2, len(met))), "n_stations": 5})
    out = robustness.h3_loyo(cfg, analysis.build_table(cfg, fd, pm, met, save=False))
    assert len(out) == 8 and out["dropped"].isna().sum() == 1
    var = robustness.h3_variants(cfg, fd, pm, met)
    assert {"baseline", "excluding 2020", "no NW-wind weighting"} <= set(var["variant"])


# ---------------------------------------------------------------- PM2.5 fallback

def test_fallback_loader_writes_openaq_layout(cfg, tmp_path):
    src = tmp_path / "csv"
    src.mkdir()
    pd.DataFrame({"Station": "Anand Vihar", "From Date": pd.date_range("2020-10-20 00:00", periods=6, freq="30min").strftime("%Y-%m-%d %H:%M:%S"),
                  "PM2.5": [100, 120, 140, 160, 180, 200]}).to_csv(src / "a.csv", index=False)
    n = pm25_fallback.load_csv_dir(cfg, src, "Station", "From Date", "PM2.5")
    assert n == 3
    f = next(cfg.raw_dir("openaq", "2020").glob("sensor_*.parquet"))
    out = pd.read_parquet(f)
    assert out["value"].tolist() == [110.0, 150.0, 190.0]
    assert out["dt_utc"].iloc[0] == "2020-10-19T18:30:00Z"      # IST input -> UTC


# ---------------------------------------------------------------- matched pixels and Aqua dose-response

def test_matched_daily_classifies_cells():
    from sfta import proxies

    rows = []
    def add(sat, lat, hour):
        rows.append({"date_ist": pd.Timestamp("2024-10-20"), "satellite": sat, "daynight": "D", "lat": lat, "lon": 75.0, "hour_ist": hour})
    # Block 30.0-30.25: S-NPP is the earlier pass (13.5 vs 14.3).
    add("SNPP", 30.011, 13.5); add("NOAA20", 30.011, 14.3)      # seen by both
    add("SNPP", 30.051, 13.5)                                    # earlier pass only
    add("NOAA20", 30.091, 14.3); add("NOAA20", 30.131, 14.3)     # later pass only (two cells)
    add("SNPP", 31.511, 13.5)                                    # block seen by one sensor only: ignored
    out = proxies.matched_daily(pd.DataFrame(rows))
    assert out.loc["2024-10-20"].to_dict() == {"early_only": 1, "late_only": 2, "both": 1}


def test_yearly_proxies_includes_new_headline_proxies(cfg):
    rng = np.random.default_rng(3)
    rows = []
    for d in pd.date_range("2023-10-01", "2023-10-31"):
        for sat, hour, n in (("SNPP", 13.6, 20), ("NOAA20", 13.6, 20), ("Aqua", 15.2, 40)):
            for lat in rng.uniform(30, 30.2, n):
                rows.append({"date_ist": d, "satellite": sat, "daynight": "D", "lat": lat, "lon": 75.0, "hour_ist": hour,
                             "frp": 10.0, "scan": 0.4, "year": 2023})
    cfg.raw["years"] = [2023, 2023]
    from sfta import proxies
    y = proxies.yearly_proxies(cfg, pd.DataFrame(rows))
    v = y.set_index("proxy")["value"]
    assert v["aqua_snpp_ratio"] == pytest.approx(2.0)
    assert v["night_count_snpp"] == 0
    assert "matched_late_only_share" in v.index
    assert not proxies.HEADLINE.count("aqua_terra_ratio")


def test_aqua_dose_response_relative_to_baseline(cfg):
    yearly = pd.DataFrame({"proxy": "aqua_snpp_ratio", "year": [2020, 2021, 2024], "value": [0.2, 0.2, 0.6], "lo": 0.1, "hi": 0.7})
    ov = pd.DataFrame({"year": [2020, 2021, 2024], "satellite": "Aqua", "median": [13.8, 13.8, 14.6]})
    out = robustness.aqua_dose_response(cfg, yearly, ov, baseline_to=2021)
    assert out.set_index("year").loc[2024, "relative"] == pytest.approx(3.0)


# ---------------------------------------------------------------- ERA5 bulk files

def _write_era5(path, kind, start, end, rng):
    import xarray as xr

    t = pd.date_range(start, end, freq="h")
    coords = {"valid_time": t, "latitude": [28.8, 28.5], "longitude": [77.0, 77.3]}
    shape = (len(t), 2, 2)
    f = lambda base, sd: (("valid_time", "latitude", "longitude"), rng.normal(base, sd, shape))
    if kind == "sfc":
        ds = xr.Dataset({"blh": f(500, 50), "u10": f(1, .2), "v10": f(1, .2), "t2m": f(300, 1), "d2m": f(290, 1), "tp": f(1e-4, 1e-5)}, coords=coords)
    else:
        ds = xr.Dataset({"u": f(5, 1), "v": f(-5, 1)}, coords=coords)
    ds.to_netcdf(path, engine="h5netcdf")


def test_era5_process_reads_year_and_bulk_files_and_drops_gaps(cfg):
    from sfta import era5

    rng = np.random.default_rng(0)
    d = cfg.raw_dir("era5")
    cfg.raw["years"] = [2020, 2023]
    for kind in ("sfc", "pl850"):
        _write_era5(d / f"{kind}_2020.nc", kind, "2020-10-01", "2020-10-05 23:00", rng)
        _write_era5(d / f"bulk_{kind}_2022_2023.nc", kind, "2022-10-01", "2023-10-03 23:00", rng)
    assert era5.bulk_covers(cfg, "sfc", 2022) and era5.bulk_covers(cfg, "sfc", 2023) and not era5.bulk_covers(cfg, "sfc", 2021)
    out = era5.process(cfg)
    assert set(out["date"].dt.year) == {2020, 2022, 2023}
    assert out["blh"].notna().all()                       # gap days between seasons are not rows
    assert out["nw_frac"].between(0, 1).all()
    assert out.loc[out["date"].dt.year == 2020, "nw_frac"].mean() > 0.9   # u>0, v<0 -> wind from the NW


def test_chunk_years():
    from sfta import era5

    assert era5.chunk_years([2021, 2018, 2019, 2020, 2022], 2) == [[2018, 2019], [2020, 2021], [2022]]


def test_era5_alt_dir_counts_as_on_disk(cfg):
    from sfta import era5

    rng = np.random.default_rng(1)
    cfg.raw["years"] = [2021, 2024]
    for kind in ("sfc", "pl850"):
        _write_era5(cfg.raw_dir("era5_alt") / f"bulk_{kind}_2021_2022.nc", kind, "2021-10-01", "2022-10-02 23:00", rng)
        _write_era5(cfg.raw_dir("era5") / f"{kind}_2024.nc", kind, "2024-10-01", "2024-10-02 23:00", rng)
    assert era5.file_years(cfg.raw_dir("era5_alt") / "bulk_sfc_2021_2022.nc") == [2021, 2022]
    assert [era5.have_year(cfg, "sfc", y) for y in (2021, 2022, 2023, 2024)] == [True, True, False, True]
    assert set(era5.process(cfg)["date"].dt.year) == {2021, 2022, 2024}


# ---------------------------------------------------------------- spillover + report

def test_spillover_control_enters_h3(cfg):
    rng = np.random.default_rng(2)
    cfg.raw["years"] = [2018, 2022]
    rows, mets, spill = [], [], []
    for y in range(2018, 2023):
        days = pd.date_range(f"{y}-09-01", f"{y}-11-30")
        n = rng.gamma(2.0, 200, len(days))
        rows.append(pd.DataFrame({"date_ist": days, "satellite": "SNPP", "n": n.round(), "frp_sum": n * 5, "daynight": "D", "state": "Punjab"}))
        spill.append(pd.DataFrame({"date_ist": days, "state": "UttarPradesh", "n": rng.poisson(50, len(days))}))
        mets.append(pd.DataFrame({"date": days, "blh": rng.normal(800, 100, len(days)), "ws10": rng.normal(2, .3, len(days)),
                                  "t2m": rng.normal(25, 2, len(days)), "rh": rng.normal(60, 5, len(days)), "tp": rng.exponential(1, len(days)),
                                  "u850": 1.0, "v850": -1.0, "nw_frac": 0.8, "blh_min": 200.0}))
    fd, met, sp = pd.concat(rows), pd.concat(mets), pd.concat(spill)
    pm = pd.DataFrame({"date": met["date"], "pm25": np.exp(4 + rng.normal(0, .2, len(met))), "n_stations": 5})
    t = analysis.build_table(cfg, fd, pm, met, save=False, spill=sp)
    assert t["spill_dl"].notna().sum() > 300
    _, _, fit = analysis.decoupling_regression(cfg, t, tag=None)
    assert "spill_dl" in fit.params.index
    cfg.raw["analysis"]["spillover_control"] = False
    assert "spill_dl" not in analysis.build_table(cfg, fd, pm, met, save=False, spill=sp)


def test_report_marks_missing_outputs_pending(cfg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sfta import report

    pd.DataFrame({"proxy": ["night_share"], "slope": [0.0041]}).to_csv(cfg.table("h1a_proxy_trends.csv"), index=False)
    fig, ax = plt.subplots()
    ax.plot([1, 2])
    fig.savefig(cfg.figure("h1a_proxy_panel.png"))
    page = report.build(cfg).read_text(encoding="utf-8")
    assert "night_share" in page and "data:image/png;base64," in page
    assert 'class="pending"' in page and "h3_beta_by_year.png" in page
