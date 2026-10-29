# SFTA — Stubble Fire Timing Analytics

Tests whether Punjab/Haryana stubble burning has shifted later in the day (past the polar-satellite
overpasses) and whether detected fire counts have decoupled from Delhi PM2.5, 2016–2026.
Full plan: the "Stubble Fire Timing Analytics Plan" doc.

Covers FIRMS proxies (H1-A), Meteosat geostationary timing (H1-B), missed-fire share (H2), decoupling tests (H3),
tracer substitution, burned area and robustness. Data notes: `docs/DATA.md`. Methods: `docs/METHODS.md`.

## Setup

```bash
python -m venv %USERPROFILE%\.venvs\sfta
%USERPROFILE%\.venvs\sfta\Scripts\pip install -e .[dev]
copy .env.example .env      # then fill in the keys
```

Keys (free sign-ups):

| Variable | Where |
| --- | --- |
| `FIRMS_MAP_KEY` | https://firms.modaps.eosdis.nasa.gov/api/map_key/ |
| `OPENAQ_API_KEY` | https://explore.openaq.org/ (account settings) |
| `CDSAPI_KEY` | https://cds.climate.copernicus.eu/ profile page; also accept the ERA5 licences once |

Raw and processed data go to `data_dir` in `config.yaml` (default `C:/sfta_data`, outside OneDrive).
Figures and tables go to `outputs/`.

## Run

```bash
python -m sfta firms-download      # ~900 API calls, cached per file; safe to re-run
python -m sfta firms-clean         # IST + solar time, state clip, confidence/type/static filters, dedup
python -m sfta firms-metrics       # daily counts per state x satellite x day/night
python -m sfta h1a                 # proxies, bootstrap CIs, Theil-Sen + Mann-Kendall, panel figure

python -m sfta pm25-download       # OpenAQ v3, CPCB stations in the Delhi box (slow: rate-limited)
python -m sfta pm25-process        # QC, station daily means, Delhi median, evening/night mean
python -m sfta era5-download       # CDS queue; one request per year per dataset
python -m sfta era5-process        # daily BLH, wind, T, RH, rain, NW-flow fraction (IST days)
python -m sfta h3                  # naive, lagged correlation, regression, counterfactual
```

Later stages (each needs extra accounts, see `.env.example`):

```bash
python -m sfta geo-download --years 2024   # LSA SAF login; ~10k small files per year
python -m sfta geo-inspect                 # print HDF5 dataset names of one file; check before parsing
python -m sfta geo-parse                   # fire pixels in Punjab/Haryana + slot table
python -m sfta geo-events                  # cluster into events, first-detection time
python -m sfta h1b                         # yearly timing stats, trends, ridgeline
python -m sfta h2                          # estimators 1 and 2 (+3 if burned-area table exists)
python -m sfta burned-area                 # Earth Engine; needs labels/burn_labels.csv
python -m sfta tracers-download && python -m sfta tracers
python -m sfta robustness                  # run after h1a, h3 (and h1b if available)
```

`all` runs only the first nine stages (FIRMS, PM2.5, ERA5, H3). `--years 2024 2025` restricts downloads. `python -m pytest` runs the tests.

## Layout

| Module | Stage |
| --- | --- |
| `config.py` | config + `.env`, data paths |
| `regions.py` | GADM Punjab/Haryana polygons |
| `firms.py` | FIRMS download, clean, daily metrics |
| `proxies.py` | H1-A proxies |
| `pm25.py` | OpenAQ download, QC, Delhi daily |
| `era5.py` | ERA5 download, daily met |
| `analysis.py` | analysis table, H3 tests |
| `stats.py` | block bootstrap, trend tests |
| `geo.py` | LSA SAF download, parse, event clustering, H1-B stats |
| `h2.py` | T_op, estimators 1 and 2 |
| `burned_area.py` | Sentinel-2 dNBR, estimator 3 (Earth Engine) |
| `tracers.py` | TROPOMI CO / MAIAC AOD, H3 substitution |
| `robustness.py` | leave-one-year-out, variants, drift adjustment |
| `pm25_fallback.py` | load CPCB / Princeton CSV exports |
| `plots.py` | figures |

## Method notes

- FIRMS `acq_time` is UTC. Both IST (+5:30) and local solar time (+5:00) are kept.
- `type` exists only in standard (SP) data. NRT rows (recent months) are filtered by a static-source
  grid built from SP `type == 2` detections.
- The fire series for H3 is S-NPP only, so NOAA-20/21 launches do not inflate counts.
- Phase-1 transport index = S-NPP fires x share of hours with 850 hPa wind from the NW sector.
- H3 regression: `log PM2.5 ~ year FE + year:fire + BLH + wind + T + RH + rain + Diwali + weekday + season spline`,
  HAC (Newey–West, 7 lags) errors. Decoupling = positive trend in the per-year fire coefficient.
