from datetime import date

import numpy as np
import pandas as pd
import pytest

from sfta import analysis, era5, firms, pm25, proxies
from sfta.config import load_config


@pytest.fixture
def cfg(tmp_path):
    c = load_config()
    c.data_dir = tmp_path / "data"
    c.outputs = tmp_path / "out"
    c.raw["years"] = [2016, 2023]
    c.raw["analysis"]["bootstrap_n"] = 200
    return c


# ---------------------------------------------------------------- FIRMS

def test_plan_requests_switches_sp_to_nrt():
    avail = pd.DataFrame({"min_date": [date(2012, 1, 1), date(2026, 6, 1)],
                          "max_date": [date(2026, 9, 3), date(2026, 10, 28)]},
                         index=["VIIRS_SNPP_SP", "VIIRS_SNPP_NRT"])
    runs = firms.plan_requests(date(2026, 9, 1), date(2026, 9, 12), "VIIRS_SNPP_SP", "VIIRS_SNPP_NRT", avail, 5)
    assert runs == [("VIIRS_SNPP_SP", date(2026, 9, 1), 3),
                    ("VIIRS_SNPP_NRT", date(2026, 9, 4), 5),
                    ("VIIRS_SNPP_NRT", date(2026, 9, 9), 4)]


def _raw_rows():
    base = dict(latitude=30.5, longitude=75.5, scan=0.4, track=0.4, version="2", bright_ti4=330, bright_ti5=290)
    rows = [
        # Two overlapping-swath copies of one S-NPP fire: keep the stronger.
        {**base, "acq_date": "2020-10-20", "acq_time": 815, "daynight": "D", "frp": 4.0, "confidence": "n", "type": 0,
         "sensor": "SNPP", "source": "VIIRS_SNPP_SP", "satellite": "N"},
        {**base, "acq_date": "2020-10-20", "acq_time": 955, "daynight": "D", "frp": 9.0, "confidence": "n", "type": 0,
         "sensor": "SNPP", "source": "VIIRS_SNPP_SP", "satellite": "N"},
        # Low confidence: dropped.
        {**base, "latitude": 30.6, "acq_date": "2020-10-20", "acq_time": 815, "daynight": "D", "frp": 3.0, "confidence": "l",
         "type": 0, "sensor": "SNPP", "source": "VIIRS_SNPP_SP", "satellite": "N"},
        # Static land source in SP data, and an NRT detection at the same spot: both dropped.
        {**base, "latitude": 31.0, "longitude": 76.0, "acq_date": "2020-10-21", "acq_time": 800, "daynight": "D", "frp": 30.0,
         "confidence": "h", "type": 2, "sensor": "SNPP", "source": "VIIRS_SNPP_SP", "satellite": "N"},
        {**base, "latitude": 31.001, "longitude": 76.001, "acq_date": "2026-10-21", "acq_time": 800, "daynight": "D", "frp": 30.0,
         "confidence": "h", "type": np.nan, "sensor": "SNPP", "source": "VIIRS_SNPP_NRT", "satellite": "N"},
        # MODIS Aqua, numeric confidence; late UTC time rolls into the next IST date.
        {**base, "latitude": 29.9, "acq_date": "2020-10-20", "acq_time": 1900, "daynight": "N", "frp": 12.0, "confidence": "55",
         "type": 0, "sensor": "MODIS", "source": "MODIS_SP", "satellite": "Aqua"},
        # MODIS below confidence threshold: dropped.
        {**base, "latitude": 29.8, "acq_date": "2020-10-20", "acq_time": 520, "daynight": "D", "frp": 12.0, "confidence": "20",
         "type": 0, "sensor": "MODIS", "source": "MODIS_SP", "satellite": "Terra"},
    ]
    return pd.DataFrame(rows)


def test_normalise_and_filter():
    cfg = load_config()
    df = firms.filter_detections(firms.normalise(_raw_rows()), cfg["firms"])
    assert len(df) == 2
    snpp = df[df["satellite"].eq("SNPP")].iloc[0]
    assert snpp["frp"] == 9.0
    assert snpp["dt_ist"] == pd.Timestamp("2020-10-20 15:25")
    assert snpp["hour_solar"] == pytest.approx(14 + 55 / 60)
    aqua = df[df["satellite"].eq("Aqua")].iloc[0]
    assert aqua["date_ist"] == pd.Timestamp("2020-10-21")


# ---------------------------------------------------------------- proxies

def test_yearly_proxies_night_share(cfg):
    rows = []
    for year, night_frac in [(2016, 0.1), (2023, 0.4)]:
        for d in pd.date_range(f"{year}-10-01", f"{year}-10-31"):
            for i in range(10):
                rows.append({"date_ist": d, "satellite": "SNPP", "daynight": "N" if i < night_frac * 10 else "D",
                             "frp": 10.0, "scan": 0.4, "hour_ist": 13.5, "year": year})
    det = pd.DataFrame(rows)
    y = proxies.yearly_proxies(cfg, det).set_index(["proxy", "year"])
    assert y.loc[("night_share_snpp", 2016), "value"] == pytest.approx(0.1)
    assert y.loc[("night_share_snpp", 2023), "value"] == pytest.approx(0.4)
    assert y.loc[("night_share_snpp", 2023), "lo"] <= 0.4 <= y.loc[("night_share_snpp", 2023), "hi"]
    assert np.isnan(y.loc[("aqua_terra_ratio", 2016), "value"])


# ---------------------------------------------------------------- PM2.5

def test_flat_run_mask():
    v = pd.Series([1, 2, 2, 2, 2, 2, 2, 3, 4])
    assert flat_list(pm25.flat_run_mask(v, 6)) == [0, 1, 1, 1, 1, 1, 1, 0, 0]


def flat_list(s):
    return s.astype(int).tolist()


def test_delhi_daily_coverage_and_night():
    pc = load_config()["pm25"]
    hours = pd.date_range("2020-10-01 00:00", "2020-10-02 23:00", freq="h")
    a = pd.DataFrame({"sensor_id": 1, "dt_ist": hours, "value": 100.0})
    b = pd.DataFrame({"sensor_id": 2, "dt_ist": hours[:10], "value": 300.0})   # too few hours for a daily mean
    out = pm25.delhi_daily(pd.concat([a, b]), pc).set_index("date")
    assert out.loc["2020-10-01", "pm25"] == 100.0
    assert out.loc["2020-10-01", "n_stations"] == 1
    assert out.loc["2020-10-01", "pm25_night"] == 100.0   # Oct 1 18:00 -> Oct 2 05:00, 12 hours


# ---------------------------------------------------------------- ERA5

def test_wind_direction_and_rh():
    assert era5.wind_from_deg(1.0, -1.0) == pytest.approx(315)   # blowing toward SE = from NW
    assert era5.relative_humidity(20.0, 20.0) == pytest.approx(100)


# ---------------------------------------------------------------- H3

def test_regression_recovers_rising_beta(cfg):
    rng = np.random.default_rng(0)
    rows = []
    years = list(range(2016, 2024))
    for y in years:
        beta = 0.1 + 0.03 * (y - 2016)
        days = pd.date_range(f"{y}-09-01", f"{y}-11-30")
        fires = rng.gamma(2.0, 200, len(days)) * np.linspace(0.2, 2, len(days))
        rows.append(pd.DataFrame({"date_ist": days, "satellite": "SNPP", "n": fires.round(), "frp_sum": fires * 5,
                                  "daynight": "D", "state": "Punjab"}))
    fd = pd.concat(rows)
    days = pd.DatetimeIndex(fd["date_ist"].unique())
    met = pd.DataFrame({"date": days, "blh": rng.normal(800, 100, len(days)), "ws10": rng.normal(2, 0.3, len(days)),
                        "t2m": rng.normal(25, 2, len(days)), "rh": rng.normal(60, 5, len(days)), "tp": rng.exponential(1.0, len(days)),
                        "u850": 1.0, "v850": -1.0, "nw_frac": 1.0, "blh_min": 200.0})
    t0 = analysis.build_table(cfg, fd, pd.DataFrame({"date": days, "pm25": 1.0, "n_stations": 1, "pm25_night": 1.0}), met)
    b = 0.1 + 0.03 * (t0["year"] - 2016)
    log_pm = 4 + b * t0["fire_dl"].fillna(0) - 0.0005 * (t0["blh"] - 800) + rng.normal(0, 0.05, len(t0))
    pm = pd.DataFrame({"date": t0["date"], "pm25": np.exp(log_pm), "n_stations": 5, "pm25_night": np.exp(log_pm)})
    t = analysis.build_table(cfg, fd, pm, met)
    beta, trend, _ = analysis.decoupling_regression(cfg, t)
    assert trend["beta_slope_per_year"] == pytest.approx(0.03, abs=0.01)
    assert trend["slope_p"] < 0.01
    resid, _ = analysis.counterfactual(cfg, t)
    assert resid.loc[resid["year"] == 2023, "mean_resid_log"].item() > 0


def _h3_fixture(cfg, hidden: bool, last_year: int = 2023):
    """Fires fall 90% in the last two seasons; PM2.5 follows them (hidden=False) or ignores the fall (hidden=True)."""
    rng = np.random.default_rng(1)
    rows = []
    for y in range(2016, last_year + 1):
        days = pd.date_range(f"{y}-09-01", f"{y}-11-30")
        fires = rng.gamma(2.0, 200, len(days)) * np.linspace(0.2, 2, len(days)) * (0.1 if y >= last_year - 1 else 1.0)
        rows.append(pd.DataFrame({"date_ist": days, "satellite": "SNPP", "n": fires.round(), "frp_sum": fires * 5,
                                  "daynight": "D", "state": "Punjab"}))
    fd = pd.concat(rows)
    days = pd.DatetimeIndex(fd["date_ist"].unique())
    met = pd.DataFrame({"date": days, "blh": rng.normal(800, 100, len(days)), "ws10": rng.normal(2, 0.3, len(days)),
                        "t2m": rng.normal(25, 2, len(days)), "rh": rng.normal(60, 5, len(days)), "tp": rng.exponential(1.0, len(days)),
                        "u850": 1.0, "v850": -1.0, "nw_frac": 1.0, "blh_min": 200.0})
    flat = pd.DataFrame({"date": days, "pm25": 1.0, "n_stations": 1, "pm25_night": 1.0})
    t0 = analysis.build_table(cfg, fd, flat, met, save=False)
    driver = t0["fire_dl"].fillna(0)
    if hidden:   # PM2.5 driven by the pre-fall fire level
        driver = driver.where(t0["year"] < last_year - 1, t0.loc[t0["year"] < last_year - 1, "fire_dl"].mean())
    log_pm = 4 + 0.15 * driver + rng.normal(0, 0.05, len(t0))
    pm = pd.DataFrame({"date": t0["date"], "pm25": np.exp(log_pm), "n_stations": 5, "pm25_night": np.exp(log_pm)})
    return analysis.build_table(cfg, fd, pm, met, save=False)


def test_hidden_fire_test_separates_cases(cfg):
    cfg.raw["analysis"]["hidden_pooled_years"] = [2022, 2023]
    lam = {}
    for hidden in (False, True):
        out = analysis.hidden_fire_test(cfg, _h3_fixture(cfg, hidden))
        lam[hidden] = out.query("form == 'log1p' and group == 'pooled 2022-2023'")["lambda"].item()
    assert abs(lam[False]) < 0.25
    assert lam[True] > 0.75


def test_functional_form_selects_a_form(cfg):
    out = analysis.functional_form(cfg, _h3_fixture(cfg, False))
    assert set(out["form"]) == {"log1p", "linear", "sqrt"}
    assert out["selected"].sum() == 1


def test_unfinished_season_is_excluded(cfg):
    from datetime import date
    year = date.today().year + 1          # season ends after today
    cfg.raw["years"] = [2016, year]
    cfg.raw["diwali"][year] = f"{year}-11-01"
    t = _h3_fixture(cfg, False, last_year=year)
    assert not t.loc[t["year"] == year, "in_season"].any()
    assert t.loc[t["year"] == year, "in_window"].any()
    assert t.loc[t["year"] == 2016, "in_season"].any()
