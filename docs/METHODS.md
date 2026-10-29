# Methods

Season: Sep 15 - Nov 30 (IST dates). Sep 1-14 is the PM2.5 baseline. Years 2016-2026; GEO events 2017-2025.

## Fire detections
UTC `acq_date`/`acq_time` become IST (+5:30) and local solar time (+5:00). Detections are clipped to Punjab and Haryana (GADM), then filtered: MODIS confidence >= 30, VIIRS nominal or high, vegetation fire (`type == 0`) in SP data, and for NRT rows a static-source mask (0.01 deg cells containing any SP `type == 2` detection). Overlapping swaths are collapsed per satellite, date, day/night and ~375 m cell, keeping the highest FRP.

## H1-A: FIRMS-only proxies
Per season, with 7-day moving-block bootstrap intervals over days and Theil-Sen / Mann-Kendall trends on the yearly series:
- **S-NPP night count** (absolute) and **night share** (night / all S-NPP); MODIS night share. Fires lit late are still burning at ~01:30. The share also rises when daytime detections fall, so the count is the cleaner measure but is noisy.
- **Low-FRP share** of S-NPP daytime detections (just lit or dying fires).
- **Aqua / S-NPP daytime ratio**. Aqua's overpass has drifted later (about 13:50 IST in 2016-2022, 15:20 in 2025), so it samples later in the afternoon each year. `h1a_aqua_dose_response.csv` expresses the ratio relative to its 2016-2022 mean (the MODIS-vs-VIIRS sensitivity gap) against Aqua's median pass hour. The orbit-drift adjustment is deliberately not applied to this proxy, since the drift is the signal.
- **Matched S-NPP / NOAA-20 blocks.** Both cross at the same local time, so the plain ratio is flat. Cells (0.01 deg) in 0.25 deg blocks where both sensors detected something on a day are compared: share of fires only the later pass sees, and share of earlier-pass fires the later pass still sees. Informative only about ignitions in the ~50 min between the passes; a supporting check.
- **Aqua / Terra ratio** (appendix): Terra sees few Oct-Nov fires and both orbits drift, so it is computed but not charted.

`robust_drift_adjusted.csv` repeats the other trends after regressing each proxy on that year's median overpass hour.

## H1-B: geostationary timing
Fire pixels from the Meteosat IODC FRP-PIXEL product are linked into events: same or 8-neighbour 0.04 deg cell, gaps up to 3 slots. The first detecting slot (centre, IST) is the ignition proxy; duration is first to last slot plus one slot. Event densities are weighted by 1 / relative number of archive slots seen in that hour, so missing files do not read as no fire. Reported per year: median, KDE peak, share after 15:00 and 17:00, with day-block bootstrap intervals. Slot time is the nominal slot start; the scan reaches northern India later in the 15-minute cycle, so absolute hours may sit up to ~15 minutes early.

Expected from the literature (checks, not inputs): peak ~13:30 (2020) to ~17:00 (2024); >90% of large Punjab fires after 15:00 in 2024-25 against 3% in 2021; shift starting 2022.

## H2: share missed
1. **GEO direct.** Share of events whose first detection is after that day's T_op, the latest afternoon FIRMS acquisition (per satellite 99th percentile of hour, maximum over satellites; days without afternoon detections use the year median).
2. **Detectability model.** A polar sensor sees an event if a pass falls between ignition and ignition + duration x multiplier (0.5, 1, 2), with daily repeating passes from that year's median FIRMS pass hours, ignition jitter within the slot and pass-time jitter. Day-block bootstrap. Reported for S-NPP only and for all polar sensors; the table headline uses all sensors, multiplier 1.
3. **Burned-area reconciliation** (pixel-level): burned cropland pixels farther than 375 m from any VIIRS detection / all burned pixels. An upper bound on the miss, since it also contains small-fire and cloud misses. Estimators 1 and 2 are different quantities (after the last pass vs not seen by any pass); the table keeps them side by side.

## H3: decoupling
Run on two PM2.5 series. The continuous US Embassy station (all seasons) is the main series, written to `outputs/usemb/`. The CPCB multi-station median (61 Delhi stations merged by site; no data 2023-2024 on OpenAQ) is secondary, written to `outputs/`; its trend rests on 2016-2021 versus 2025 only. Seasons lacking November data are dropped, and so is any season not yet finished (`analysis.exclude_unfinished_season`), so a partial current season cannot bias the beta trend.

Daily Delhi PM2.5 = median over stations of station daily means (>= 75% of hours; values <= 0 or > 999 and runs of 6+ identical hours removed). Fire index = S-NPP detections x share of hours with 850 hPa wind from 270-360 deg, lagged 1-2 days and averaged within season. Model:

`log PM2.5 = year FE + beta_year x fire + BLH + wind + T + RH + rain + Diwali(+/-2 d) + weekday + B-spline(day of season) + spill + e`

`spill` is log(1 + S-NPP fires in Uttar Pradesh and Madhya Pradesh), lagged and averaged like the fire index but not wind-weighted, since that smoke arrives on easterly flow. It keeps fires outside Punjab and Haryana from reading as decoupling (`analysis.spillover_control`).

HAC (Newey-West, 7 lags) errors. Decoupling is a positive trend in beta_year, tested as a linear contrast with centred years. The counterfactual fits fire + meteorology on 2016-2021 and reports the mean log residual for each season. Unlike the year-FE regression, it has no season intercepts, so a season-wide undercount of fires shows up as a positive residual.

**Hidden-fire test** (`h3_hidden_fire_test.csv`). `hidden` is the residual expected if smoke had stayed at the 2016-2021 mean fire level while detected counts fell. lambda = residual / hidden: 0 means detected counts explain PM2.5, 1 means the fall in detected fires was not matched by a fall in PM2.5. Reported per test season and pooled over 2023-2025, for each fire-term form.

**Functional form** (`h3_functional_form.csv`). The model is only informative about hidden fires through the fire-term form, because test-season fire counts lie below the training range. log(1+fires), linear and square-root terms are compared by leave-one-season-out prediction error on the training seasons only; the form is chosen before looking at the test seasons.

**Naive ratio.** `excess_per_1000_fires` in `h3_naive_yearly.csv` is descriptive only. The seasonal PM2.5 excess (against 1-14 Sep) stays roughly flat while fires fall by an order of magnitude, so the ratio rises mechanically whether or not fires were hidden. `fires_rel_train` and `excess_rel_train` show numerator and denominator separately. Lagged correlations use first differences of log(1 + fires) and log PM2.5 with block-bootstrap intervals. Tracer substitution replaces fires with TROPOMI CO or MAIAC AOD; the same model is also run with evening/night and daytime PM2.5 as outcomes.

## Robustness
Leave-one-year-out for the H1 trends and the H3 beta trend; H3 variants (lags 0-1 and 1-3, no wind weighting, all polar sensors, Aqua only, without the UP/MP spillover control, without 2020, without the Diwali window). With about 11 seasons, effect sizes and intervals are the result; p-values are secondary.

## Limits
- Geostationary pixels (~4-5 km over Punjab) detect only larger or hotter fires, so GEO gives timing, not counts.
- GEO duration is a lower bound (detection threshold, cloud), hence the duration multipliers.
- Brick kilns and other non-agricultural fires are not masked in the GEO series.
- Burned-area labels are hand-made; the dNBR threshold is only as good as them.
