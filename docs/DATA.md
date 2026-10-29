# Data sources

Raw data lives in `data_dir` (default `C:/sfta_data/raw/<source>/`), never in the repo. Stages skip files that already exist, so downloads can be re-run.

| Source | Stage | Access | Raw layout | Caveats |
| --- | --- | --- | --- | --- |
| NASA FIRMS (MODIS, VIIRS S-NPP / NOAA-20 / NOAA-21) | `firms-download` | `FIRMS_MAP_KEY`; area API, 5 days per call | `firms/<SENSOR>/<SOURCE>_<first day>_<n>.csv` | Standard (`*_SP`) archive used where it covers the day, near-real-time (`*_NRT`) after. Coverage is read from `data_availability` at run time. `type` exists only in SP data. NOAA-20 SP starts 2018-04, NOAA-21 SP 2023-03. |
| OpenAQ v3 (CPCB stations) | `pm25-download` | `OPENAQ_API_KEY` | `openaq/<year>/sensor_<id>.parquet`, `sensors.csv` | Mixes providers; only locations whose provider matches `pm25.provider_pattern` are used. Check `sensors.csv` after the first run. |
| CSV exports (CPCB CCR, Princeton 2015-2019) | `pm25_fallback.load_csv_dir` | manual download | same layout, negative sensor ids | Column names must be supplied; layouts not verified. |
| NASA FIRMS S-NPP, UP + MP | `spill-download`, `spill-process` | `FIRMS_MAP_KEY` | `firms_spill/SNPP/...` | Spillover covariate for H3 only; same cleaning as the main FIRMS series. |
| ERA5 single levels + 850 hPa | `era5-queue` (or `era5-download`) | `CDSAPI_KEY`, licences accepted | `era5/sfc_<year>.nc`, `pl850_<year>.nc`, `bulk_<kind>_<y0>_<y1>.nc`; also `era5_alt/` | Delhi box mean, 3-hourly; UTC converted to IST days. `era5_alt/` holds files fetched by side scripts and is read like `era5/`. `era5-status` lists CDS jobs and missing years. |
| LSA SAF MSG-IODC FRP-PIXEL | `geo-download` | `LSASAF_USER` / `LSASAF_PASSWORD` (free registration) | `geo/<year>/HDF5_..._ListProduct_..._<YYYYMMDDHHMM>` | Demonstration product, 2017 to present, 15 min. Folder listing is public, files need login. Dataset names inside the HDF5 are matched by name; run `geo-inspect` on the first download. |
| GADM 4.1 India level 1 | automatic | public | `boundaries/gadm41_IND_1.json` | Punjab + Haryana polygons. |
| Sentinel-2 SR, ESA WorldCover, FAO GAUL | `burned-area` | Earth Engine, `EE_PROJECT` | server side | Needs `labels/burn_labels.csv` (lon, lat, year, burned). 2016-2017 flagged low confidence. |
| TROPOMI CO, MAIAC AOD | `tracers-download` | Earth Engine | server side | 2018 onward. |

## Processed tables (`data_dir/processed/`)

| File | Rows | Written by |
| --- | --- | --- |
| `firms_detections.parquet` | one per cleaned detection, IST + solar time, state | `firms-clean` |
| `firms_daily.parquet` | date x state x satellite x day/night | `firms-metrics` |
| `pm25_delhi_daily.parquet` | Delhi daily PM2.5, evening/night and daytime means | `pm25-process` |
| `era5_delhi_daily.parquet` | daily BLH, wind, T, RH, rain, NW-flow share | `era5-process` |
| `geo_pixels.parquet`, `geo_slots.parquet` | fire pixels; every slot that has a file | `geo-parse` |
| `geo_events.parquet` | clustered fire events with first-detection time | `geo-events` |
| `firms_spill_daily.parquet` | date x state S-NPP fire counts, Uttar Pradesh and Madhya Pradesh | `spill-process` |
| `analysis_table.parquet` | one row per day, all covariates | `h3` |
| `tracers_daily.parquet` | TROPOMI CO, MAIAC AOD | `tracers-download` |

Outputs: `outputs/tables/*.csv`, `outputs/figures/*.png`, and `outputs/report.html` (all tables and figures on one page, written by `report`; not committed).

## Known gaps

- **OpenAQ has no CPCB data from ~Nov 2022 to Dec 2024** (checked 2026-10-29: 0 hourly rows for 2023 and 2024, 2022 stops in October). That removes the two most important post-shift seasons from the CPCB multi-station median. The US Embassy monitor in New Delhi (OpenAQ sensors 5398 and 23534) is continuous 2016-2026 and is kept as a second series (`pm25_ref_daily.parquet`); it reads ~9% below the CPCB median where both exist and the season means move together. H3 runs on both (`outputs/` for CPCB, `outputs/usemb/` for the US Embassy). The CPCB series can be completed from CCR portal exports with `pm25_fallback.load_csv_dir`.
- Seasons without November PM2.5 (fewer than `analysis.min_november_days` valid days) are dropped from H3 so partial series do not bias the mean.

- 2026 is a partial season (near-real-time data, post-season windows incomplete). Re-run after Nov 30.
- No GEO cloud-fraction covariate yet.
- Orbit drift is handled by year fixed effects in the regression and by `robust_drift_adjusted.csv` for the FIRMS proxies.
