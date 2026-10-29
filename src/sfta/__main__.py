"""Command line: python -m sfta <stage> [--years 2019 2020]."""
from __future__ import annotations

import argparse

import pandas as pd

from . import analysis, burned_area, era5, firms, geo, h2, pm25, plots, proxies, report, robustness, tracers
from .config import load_config


def run_h1a(cfg):
    yearly = proxies.yearly_proxies(cfg)
    trends = proxies.proxy_trends(cfg, yearly)
    ov = proxies.overpass_times(cfg)
    plots.proxy_panel(cfg, yearly, trends)
    plots.aqua_dose_response(cfg, robustness.aqua_dose_response(cfg, yearly, ov))
    print(trends.to_string(index=False))


def run_geo_inspect(cfg):
    files = sorted(cfg.raw_dir("geo").glob("*/*"))
    if not files:
        raise SystemExit("No raw GEO files. Run: python -m sfta geo-download --years 2024")
    print(geo.inspect_file(files[len(files) // 2]))


def run_h1b(cfg):
    ev = pd.read_parquet(cfg.processed("geo_events.parquet"))
    slots = pd.read_parquet(cfg.processed("geo_slots.parquet"))
    yearly = geo.yearly_timing(cfg, ev, slots)
    trends = geo.timing_trends(cfg, yearly)
    plots.ridgeline(cfg, ev, geo.event_weights(ev, geo.observed_slots_per_hour(slots, cfg)))
    print(yearly[["year", "n", "median", "peak", *[c for c in yearly if c.startswith("share_after") and not c.endswith(("lo", "hi"))]]].to_string(index=False))
    print(trends.to_string(index=False))


def run_h2(cfg):
    table = h2.run(cfg)
    plots.h2_panel(cfg, table)
    print(table.round(3).to_string(index=False))


# usemb is the main series (continuous, one site); cpcb is secondary (gaps in 2022-2024, station set varies).
SERIES = {"usemb": "pm25_ref_daily.parquet", "cpcb": "pm25_delhi_daily.parquet"}


def run_h3(cfg):
    """H3 on both PM2.5 series: the continuous US Embassy station (main) and the CPCB station median (secondary)."""
    from dataclasses import replace

    done = {}
    for name, f in SERIES.items():
        if not cfg.processed(f).exists():
            print(f"-- skipping {name}: {f} missing")
            continue
        print(f"##### PM2.5 series: {name}")
        c = cfg if name == "cpcb" else replace(cfg, outputs=cfg.outputs / name)
        t = _h3_one(c, pd.read_parquet(c.processed(f)))
        t.to_parquet(cfg.processed(f"analysis_table_{name}.parquet"), index=False)
        done[name] = t
    # robustness and tracers read analysis_table.parquet: use the continuous series when it exists.
    main = done.get("usemb", done.get("cpcb"))
    if main is not None:
        main.to_parquet(cfg.processed("analysis_table.parquet"), index=False)


def _h3_one(cfg, pm):
    t = analysis.build_table(cfg, pm=pm, save=False)
    print(analysis.naive_yearly(cfg, t).to_string(index=False))
    plots.corr_heatmap(cfg, analysis.lagged_correlations(cfg, t))
    beta, trend, _ = analysis.decoupling_regression(cfg, t)
    plots.beta_plot(cfg, beta, trend)
    print(beta.to_string(index=False))
    print(trend)
    resid, _ = analysis.counterfactual(cfg, t)
    plots.counterfactual_plot(cfg, resid)
    print(resid.to_string(index=False))
    print(analysis.functional_form(cfg, t).round(4).to_string(index=False))
    print(analysis.hidden_fire_test(cfg, t).round(3).to_string(index=False))
    return t


def run_tracers(cfg):
    t = pd.read_parquet(cfg.processed("analysis_table.parquet"))
    print(tracers.substitution(cfg, t).round(4).to_string(index=False))


def run_burned(cfg):
    ba = burned_area.run(cfg)
    plots.burned_vs_detected(cfg, ba)


def run_robust(cfg):
    from dataclasses import replace

    # H3 checks per PM2.5 series, written next to that series' H3 outputs.
    for name, f in SERIES.items():
        if not cfg.processed(f"analysis_table_{name}.parquet").exists():
            continue
        print(f"##### PM2.5 series: {name}")
        c = cfg if name == "cpcb" else replace(cfg, outputs=cfg.outputs / name)
        t = pd.read_parquet(cfg.processed(f"analysis_table_{name}.parquet"))
        print(robustness.h3_loyo(c, t).round(4).to_string(index=False))
        print(robustness.h3_variants(c, pm=pd.read_parquet(cfg.processed(f))).round(4).to_string(index=False))
    yearly = pd.read_csv(cfg.table("h1a_proxies_yearly.csv"))
    ov = pd.read_csv(cfg.table("overpass_times_ist.csv"))
    print(robustness.drift_adjusted_trends(cfg, yearly, ov).round(4).to_string(index=False))
    try:
        timing = pd.read_csv(cfg.table("h1b_timing_yearly.csv"))
    except FileNotFoundError:
        timing = pd.DataFrame(columns=["year"])
    robustness.h1_loyo(cfg, timing, yearly)


STAGES = {
    "firms-download": lambda c, y: firms.download(c, y),
    "firms-clean": lambda c, y: firms.clean(c),
    "firms-metrics": lambda c, y: firms.daily_metrics(c),
    "h1a": lambda c, y: run_h1a(c),
    "pm25-download": lambda c, y: pm25.download(c, y),
    "pm25-process": lambda c, y: (pm25.process(c), pm25.process_reference(c)),
    "era5-download": lambda c, y: era5.download(c, y),
    "era5-queue": lambda c, y: era5.download_queue(c, y),
    "era5-status": lambda c, y: era5.status(c),
    "era5-process": lambda c, y: era5.process(c),
    "spill-download": lambda c, y: firms.spill_download(c, y),
    "spill-process": lambda c, y: firms.spill_daily(c),
    "h3": lambda c, y: run_h3(c),
    "geo-download": lambda c, y: geo.download(c, y),
    "geo-inspect": lambda c, y: run_geo_inspect(c),
    "geo-parse": lambda c, y: geo.parse(c),
    "geo-events": lambda c, y: geo.build_events(c),
    "h1b": lambda c, y: run_h1b(c),
    "h2": lambda c, y: run_h2(c),
    "burned-area": lambda c, y: run_burned(c),
    "tracers-download": lambda c, y: tracers.download(c),
    "tracers": lambda c, y: run_tracers(c),
    "robustness": lambda c, y: run_robust(c),
    "report": lambda c, y: report.build(c),
}
# `all` runs the stages that need only FIRMS, OpenAQ and ERA5 keys; the rest are run by name.
CORE = ["firms-download", "firms-clean", "firms-metrics", "h1a", "spill-download", "spill-process",
        "pm25-download", "pm25-process", "era5-download", "era5-process", "h3", "report"]


def main():
    p = argparse.ArgumentParser(prog="sfta")
    p.add_argument("stage", choices=[*STAGES, "all"])
    p.add_argument("--years", type=int, nargs="*", help="restrict downloads to these years")
    p.add_argument("--config", help="path to config.yaml")
    a = p.parse_args()
    cfg = load_config(a.config)
    for name in (CORE if a.stage == "all" else [a.stage]):
        print(f"== {name}")
        STAGES[name](cfg, a.years)


if __name__ == "__main__":
    main()
