"""Analysis table and H3 decoupling tests (naive, lagged correlation, regression, counterfactual)."""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from .config import Config
from .stats import boot_ci

MET = "blh_km + ws10 + t2m + rh + tp + diwali + C(dow) + bs(doy, df=4)"


def controls(t: pd.DataFrame) -> str:
    """Meteorology and calendar terms, plus the UP/MP spillover fires when the table carries them."""
    return MET + (" + spill_dl" if "spill_dl" in t else "")


# ---------------------------------------------------------------- table

def build_table(cfg: Config, firms_daily=None, pm=None, met=None, save: bool = True, spill=None) -> pd.DataFrame:
    firms_daily = firms_daily if firms_daily is not None else pd.read_parquet(cfg.processed("firms_daily.parquet"))
    pm = pm if pm is not None else pd.read_parquet(cfg.processed("pm25_delhi_daily.parquet"))
    met = met if met is not None else pd.read_parquet(cfg.processed("era5_delhi_daily.parquet"))
    use_spill = cfg["analysis"].get("spillover_control", False)
    if spill is None and use_spill and cfg.processed("firms_spill_daily.parquet").exists():
        spill = pd.read_parquet(cfg.processed("firms_spill_daily.parquet"))

    days = pd.DatetimeIndex(np.concatenate([
        pd.date_range(cfg.season_bounds(y, "baseline")[0], cfg.season_bounds(y, "season")[1]) for y in cfg.years
    ]), name="date")
    sensor = cfg["analysis"]["fire_sensor"]
    f = firms_daily.rename(columns={"date_ist": "date"})
    one = (f if sensor == "ALL" else f[f["satellite"].eq(sensor)]).groupby("date").agg(fires=("n", "sum"), frp=("frp_sum", "sum"))
    allsat = f.groupby("date").agg(fires_all=("n", "sum"))
    t = pd.DataFrame(index=days).join(one).join(allsat)
    t[["fires", "frp", "fires_all"]] = t[["fires", "frp", "fires_all"]].fillna(0)
    t = t.join(pm.set_index("date")).join(met.set_index("date"))

    t["year"] = t.index.year
    t["doy"] = (t.index - pd.to_datetime(t["year"].astype(str) + "-09-01")).dt.days
    t["dow"] = t.index.day_of_week
    dw = {int(y): pd.Timestamp(d) for y, d in cfg["diwali"].items()}
    t["diwali"] = [int(abs((d - dw[d.year]).days) <= 2) if d.year in dw else 0 for d in t.index]
    t["blh_km"] = t["blh"] / 1000
    # Phase-1 transport weighting: fires count toward Delhi in proportion to NW 850 hPa flow that day.
    t["fire_idx"] = t["fires"] * (t["nw_frac"] if cfg["analysis"].get("weight_nw", True) else 1.0)
    t["log_fire"] = np.log1p(t["fire_idx"])
    t["log_pm"] = np.log(t["pm25"])
    for col in ("pm25_night", "pm25_day"):
        if col in t:
            t["log_" + col] = np.log(t[col].where(t[col] > 0))
    lagged = [t.groupby("year")["log_fire"].shift(k) for k in cfg["analysis"]["fire_lags"]]
    t["fire_dl"] = pd.concat(lagged, axis=1).mean(axis=1, skipna=False)
    # Alternative functional forms of the same lagged index, for the form comparison (functional_form).
    for col, f_ in (("fire_lin_dl", t["fire_idx"] / 1000), ("fire_sqrt_dl", np.sqrt(t["fire_idx"]))):
        t[col] = pd.concat([f_.groupby(t["year"]).shift(k) for k in cfg["analysis"]["fire_lags"]],
                           axis=1).mean(axis=1, skipna=False)
    if use_spill and spill is not None:
        # Same lags as the Punjab/Haryana index; not wind-weighted (UP/MP smoke arrives on easterly flow).
        t["log_spill"] = np.log1p(t.join(spill.groupby("date_ist")["n"].sum().rename("spill"))["spill"].fillna(0))
        t["spill_dl"] = pd.concat([t.groupby("year")["log_spill"].shift(k) for k in cfg["analysis"]["fire_lags"]],
                                  axis=1).mean(axis=1, skipna=False)
    a, b = cfg["season"]["start"], cfg["season"]["end"]
    md = t.index.strftime("%m-%d")
    t["in_season"] = (md >= a) & (md <= b)
    t["in_window"] = t["in_season"]          # calendar season; in_season below also drops unusable seasons
    # An unfinished season (today is before its last day) would bias seasonal means and the beta trend.
    today = pd.Timestamp(date.today())
    unfinished = [int(y) for y in t["year"].unique() if pd.Timestamp(cfg.season_bounds(int(y), "season")[1]) >= today]
    if unfinished and cfg["analysis"].get("exclude_unfinished_season", True):
        print(f"analysis table: unfinished seasons dropped: {sorted(unfinished)}")
        t.loc[t["year"].isin(unfinished), "in_season"] = False
    # A finished season needs November PM2.5: partial series (e.g. OpenAQ CPCB stops Oct 2022) would bias the mean low.
    nov = t[(t.index.month == 11) & t["pm25"].notna()].groupby("year").size()
    short = [y for y in t["year"].unique() if y not in unfinished and nov.get(y, 0) < cfg["analysis"].get("min_november_days", 15)]
    if short:
        print(f"analysis table: seasons dropped for missing November PM2.5: {sorted(short)}")
        t.loc[t["year"].isin(short), "in_season"] = False
    t = t.reset_index()
    if save:
        t.to_parquet(cfg.processed("analysis_table.parquet"), index=False)
    return t


# ---------------------------------------------------------------- H3

def naive_yearly(cfg: Config, t: pd.DataFrame) -> pd.DataFrame:
    s = t[t["in_season"]].groupby("year").agg(fires=("fires", "sum"), fires_all=("fires_all", "sum"), pm25_season=("pm25", "mean"))
    base = t[~t["in_window"]].groupby("year")["pm25"].mean().rename("pm25_baseline")
    out = s.join(base)
    out["pm25_excess"] = out["pm25_season"] - out["pm25_baseline"]
    # Descriptive only: when fires collapse and the excess stays roughly flat, this ratio rises mechanically,
    # whether or not fires were hidden. The two columns below show the numerator and denominator separately.
    out["excess_per_1000_fires"] = 1000 * out["pm25_excess"] / out["fires"].replace(0, np.nan)
    a, b = cfg["analysis"]["train_years"]
    ref = out.loc[out.index.to_series().between(a, b)]
    out["fires_rel_train"] = out["fires"] / ref["fires"].mean()
    out["excess_rel_train"] = out["pm25_excess"] / ref["pm25_excess"].mean()
    out = out.reset_index()
    out.to_csv(cfg.table("h3_naive_yearly.csv"), index=False)
    return out


def _corr(x, y):
    ok = np.isfinite(x) & np.isfinite(y)
    return np.corrcoef(x[ok], y[ok])[0, 1] if ok.sum() > 3 else np.nan


def lagged_correlations(cfg: Config, t: pd.DataFrame) -> pd.DataFrame:
    """Per year and lag: corr of first-differenced log fires (t-k) with first-differenced log PM2.5 (t)."""
    ac = cfg["analysis"]
    rows = []
    for year, g in t[t["in_season"]].groupby("year"):
        dfire = np.log1p(g["fires"]).diff()
        dpm = g["log_pm"].diff()
        for k in ac["corr_lags"]:
            x, y = dfire.shift(k).to_numpy(), dpm.to_numpy()
            est, lo, hi = boot_ci(_corr, [x, y], ac["bootstrap_n"], ac["block_days"], ac["seed"] + year * 10 + k)
            rows.append({"year": year, "lag": k, "r": est, "lo": lo, "hi": hi})
    out = pd.DataFrame(rows)
    out.to_csv(cfg.table("h3_lagged_correlations.csv"), index=False)
    return out


def _season_frame(t: pd.DataFrame, dep: str = "log_pm", fire: str = "fire_dl", years=None) -> pd.DataFrame:
    cols = [dep, fire, "blh_km", "ws10", "t2m", "rh", "tp", "diwali", "dow", "doy", "year"]
    cols += ["spill_dl"] if "spill_dl" in t else []
    d = t[t["in_season"]].dropna(subset=cols).copy()
    d["year"] = d["year"].astype(int)
    if years is not None:
        d = d[d["year"].isin(list(years))]
    return d


def decoupling_regression(cfg: Config, t: pd.DataFrame, dep: str = "log_pm", fire: str = "fire_dl",
                          years=None, tag: str | None = ""):
    """dep ~ year FE + year x fire + met; HAC SEs. Returns (per-year beta table, trend dict, fit).

    tag=None skips writing files; any other string is appended to the output file names.
    """
    d = _season_frame(t, dep, fire, years)
    fit = smf.ols(f"{dep} ~ C(year) + C(year):{fire} + {controls(d)}", data=d).fit(cov_type="HAC", cov_kwds={"maxlags": 7})
    names = [n for n in fit.params.index if n.endswith(f":{fire}")]
    yrs = [int(n.split("[")[1].split("]")[0]) for n in names]
    beta = pd.DataFrame({
        "year": yrs,
        "beta": fit.params[names].to_numpy(),
        "se": fit.bse[names].to_numpy(),
        "n_days": [int((d["year"] == y).sum()) for y in yrs],
    })
    beta["lo"], beta["hi"] = beta["beta"] - 1.96 * beta["se"], beta["beta"] + 1.96 * beta["se"]

    # Trend in beta across years: linear contrast sum_y w_y * beta_y with w = centred year, HAC covariance.
    w = beta["year"] - beta["year"].mean()
    w = w / (w ** 2).sum()
    idx = {n: i for i, n in enumerate(fit.params.index)}
    r = np.zeros(len(fit.params))
    for n, wi in zip(names, w):
        r[idx[n]] = wi
    ct = fit.t_test(r)
    eye = np.eye(len(fit.params))
    equal = fit.wald_test(np.array([eye[idx[a]] - eye[idx[names[0]]] for a in names[1:]]), scalar=True)
    trend = {
        "beta_slope_per_year": float(np.asarray(ct.effect).ravel()[0]),
        "slope_se": float(np.asarray(ct.sd).ravel()[0]),
        "slope_p": float(np.asarray(ct.pvalue).ravel()[0]),
        "equal_beta_wald_p": float(equal.pvalue),
        "n_obs": int(fit.nobs),
        "r2": float(fit.rsquared),
    }
    if tag is not None:
        beta.to_csv(cfg.table(f"h3_beta_by_year{tag}.csv"), index=False)
        pd.DataFrame([trend]).to_csv(cfg.table(f"h3_beta_trend{tag}.csv"), index=False)
    return beta, trend, fit


FORMS = {"log1p": "fire_dl", "linear": "fire_lin_dl", "sqrt": "fire_sqrt_dl"}


def counterfactual(cfg: Config, t: pd.DataFrame):
    out, fit, _, _ = _counterfactual(cfg, t)
    return out, fit


def _counterfactual(cfg: Config, t: pd.DataFrame, fire: str = "fire_dl", save: bool = True):
    """Fit fires + met on pre-shift years; mean residual (observed - predicted log PM2.5) per year."""
    ac = cfg["analysis"]
    d = _season_frame(t, fire=fire)
    a, b = ac["train_years"]
    train = d[d["year"].between(a, b) & ~d["year"].isin(ac["exclude_from_train"])]
    fit = smf.ols(f"log_pm ~ {fire} + {controls(train)}", data=train).fit(cov_type="HAC", cov_kwds={"maxlags": 7})
    d["pred"] = fit.predict(d)
    d["resid"] = d["log_pm"] - d["pred"]
    rows = []
    for year, g in d.groupby("year"):
        r = g["resid"].to_numpy()
        est, lo, hi = boot_ci(np.mean, [r], ac["bootstrap_n"], ac["block_days"], ac["seed"] + year)
        rows.append({"year": year, "train": a <= year <= b and year not in ac["exclude_from_train"],
                     "mean_resid_log": est, "lo": lo, "hi": hi,
                     "pct_above_pred": 100 * (np.exp(est) - 1), "n_days": len(r)})
    out = pd.DataFrame(rows)
    if save:
        out.to_csv(cfg.table("h3_counterfactual_residuals.csv"), index=False)
        d[["date", "year", "log_pm", "pred", "resid"]].to_parquet(cfg.processed("h3_counterfactual_daily.parquet"), index=False)
    return out, fit, d, train


def functional_form(cfg: Config, t: pd.DataFrame) -> pd.DataFrame:
    """Compare log1p / linear / sqrt fire terms on the training seasons by leave-one-season-out prediction.

    Only training seasons enter, so the choice of form does not depend on the seasons being tested.
    """
    ac = cfg["analysis"]
    a, b = ac["train_years"]
    rows = []
    for form, col in FORMS.items():
        d = _season_frame(t, fire=col)
        train = d[d["year"].between(a, b) & ~d["year"].isin(ac["exclude_from_train"])]
        rhs = f"{col} + {controls(train)}"
        fit = smf.ols(f"log_pm ~ {rhs}", data=train).fit(cov_type="HAC", cov_kwds={"maxlags": 7})
        err, year_err = [], []
        for y in sorted(train["year"].unique()):
            tr, te = train[train["year"] != y], train[train["year"] == y]
            e = (te["log_pm"] - smf.ols(f"log_pm ~ {rhs}", data=tr).fit().predict(te)).to_numpy()
            err.append(e)
            year_err.append(e.mean())
        rows.append({"form": form, "coef": fit.params[col], "se": fit.bse[col], "t": fit.tvalues[col],
                     "r2": fit.rsquared, "aic": smf.ols(f"log_pm ~ {rhs}", data=train).fit().aic,
                     "cv_rmse_daily": float(np.sqrt(np.mean(np.concatenate(err) ** 2))),
                     "cv_rmse_season_mean": float(np.sqrt(np.mean(np.square(year_err)))), "n_obs": len(train)})
    out = pd.DataFrame(rows)
    out["selected"] = out["cv_rmse_daily"].eq(out["cv_rmse_daily"].min())
    out.to_csv(cfg.table("h3_functional_form.csv"), index=False)
    return out


def hidden_fire_test(cfg: Config, t: pd.DataFrame) -> pd.DataFrame:
    """Does PM2.5 follow the fall in detected fires (counts valid) or stay at its pre-shift level (fires hidden)?

    For each form and test season, residual = observed - predicted log PM2.5 (model fitted on training seasons
    from detected fires). `hidden` is the residual expected if smoke stayed at the training-season mean fire level.
    lambda = residual / hidden: 0 means detected counts explain PM2.5, 1 means the whole fall in detected fires
    was not matched by a fall in PM2.5. Intervals: block bootstrap of the daily residuals; `hidden_lo/hi` carry the
    coefficient's standard error.
    """
    ac = cfg["analysis"]
    pooled = tuple(ac.get("hidden_pooled_years", [2023, 2025]))
    rows = []
    for form, col in FORMS.items():
        _, fit, d, train = _counterfactual(cfg, t, fire=col, save=False)
        beta, se, ref = fit.params[col], fit.bse[col], train[col].mean()
        groups = {str(y): [y] for y in sorted(d["year"].unique()) if y not in set(train["year"])}
        groups[f"pooled {pooled[0]}-{pooled[1]}"] = [y for y in sorted(d["year"].unique()) if pooled[0] <= y <= pooled[1]]
        for label, yrs in groups.items():
            g = d[d["year"].isin(yrs)]
            if g.empty:
                continue
            gap = ref - g[col].mean()                      # fall in the fire index relative to the training level
            hidden = beta * gap
            est, lo, hi = boot_ci(np.mean, [g["resid"].to_numpy()], ac["bootstrap_n"], ac["block_days"], ac["seed"] + len(yrs))
            ok = hidden > 0
            rows.append({"form": form, "group": label, "n_days": len(g), "mean_resid_log": est, "lo": lo, "hi": hi,
                         "hidden": hidden, "hidden_lo": (beta - 1.96 * se) * gap, "hidden_hi": (beta + 1.96 * se) * gap,
                         "lambda": est / hidden if ok else np.nan, "lambda_lo": lo / hidden if ok else np.nan,
                         "lambda_hi": hi / hidden if ok else np.nan,
                         "ci_includes_0": bool(lo <= 0 <= hi), "ci_includes_hidden": bool(lo <= hidden <= hi)})
    out = pd.DataFrame(rows)
    out.to_csv(cfg.table("h3_hidden_fire_test.csv"), index=False)
    return out
