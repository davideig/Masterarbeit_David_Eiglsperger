# Operational data specification for the cutoff models

Status: 2026-09-28. All four models specified; reserve timing (5.5) measured;
07:00 uses the 03 UTC run (decided 2026-09-28).

## 1. General rules

- **Cutoffs and run start.** Forecasts are created for the cutoffs 07:00, 08:00,
  09:00, 10:00, 11:00, and 12:00 local time (Europe/Berlin) on day d-1. The
  cutoff is the Energy-Arena submission deadline. Each model run starts
  **20 minutes before** its cutoff (06:40, 07:40, 08:40, 09:40, 10:40, 11:40).
  An input may only be used if it is available at the **run start**.
- **Season-independent.** All choices below are valid in summer (CEST, UTC+2)
  and winter (CET, UTC+1). Weather runs are defined in UTC and do not shift
  with daylight saving time, so the schedule follows the summer case, which is
  the binding one. In winter it is merely conservative.
- **Fixed weather run, not "latest".** Each cutoff requests one specific ICON-D2
  run explicitly. Do not use "latest available", otherwise the live model can
  receive a different run than it was trained and backtested with.
- **Missing or incomplete run.** If the fixed run is not available at the run
  start, **or is available but its required variables are null**, use the
  previous run of the same model (e.g. 03 UTC instead of 06 UTC). An earlier
  run is always admissible. Store the data under the requested run's history
  with a flag naming the run actually used. Check for null values explicitly:
  Open-Meteo returned the ICON-D2 06 UTC runs of 2026-05-13, 07-08, and 07-28
  with all wind variables null while temperature and pressure were present, and
  a run-level availability check does not catch this. Variables that are always
  null for a model (see 4.2) are excluded from this check.
- **Publication delays used.**

  | Source | Available |
  |---|---|
  | ICON-D2 via Open-Meteo | about 1.4 h after run start (00 UTC: 03:24 CEST, 03 UTC: 06:24 CEST, 06 UTC: 09:24 CEST) |
  | ENTSO-E realized load | at most 1 h after the end of each quarter-hour (Regulation 543/2013, Art. 6(1)(a)) |
  | ENTSO-E day-ahead load forecast | observed around 10:30 on d-1 (regulatory deadline 10:00) |

- **Realized-data limit.** Only quarter-hours that **end at least 1 h before the
  run start** are guaranteed to be published. This limit applies equally to
  the morning features, to the training labels, and to the comparison window
  on d-8.

### 1.1 Storage and update rules (all models)

| Data | Rule |
|---|---|
| Static data (MaStR capacity, cluster assignment, population weights, state-to-TSO mapping) | Stored once as a **fixed snapshot** and not updated. Capacity added after the snapshot is not reflected in the capacity inputs. The rolling bias correction partly absorbs the resulting drift |
| Weather | Stored as aggregated features, not as GRIB files. **A separate history per ICON-D2 run** (00, 03, 06 UTC). Each run appends the features for the new delivery day. A model trains only on the history of the run it predicts with |
| Realized values (load, generation) | Stored in a local history. On each run, **re-fetch and overwrite the last 7-14 days**, because ENTSO-E revises published values (first estimate, later measured data). Then append the new data |
| Cutoff limits | Applied **when data is used** (features, labels, bias correction), not when it is stored. The stored history always holds everything published so far |

**Transition for the early run (03 UTC).** Until a run has its own
history covering the full training window (90 days solar, 180 days wind,
224 days load), a model using that run is trained partly on another run's
history and predicts with its own run. This train/predict mismatch disappears
once enough own history exists. To shorten the transition, back-fill the
03 UTC run from the Open-Meteo single-run archive (about six months available).
The DWD GRIB path cannot be back-filled, since DWD serves only live runs. Do not
seed the history of one run by copying another run's history.

## 2. Load model

### 2.1 Inputs used at every cutoff

| Input | Source | Operational handling |
|---|---|---|
| Calendar | computed locally | weekday, 3 annual harmonics, public holidays, bridge days (holiday library). No fetch |
| Population weights | Eurostat GISCO grid, countries DE and LU | static file (`weather_cluster_population_weights_c25.csv`), fixed snapshot (see 1.1) |
| Cluster assignment | ICON-D2 grid, 25 clusters | static file (`icon_d2_clustering_c25.parquet`) |
| Load lags | ENTSO-E realized load (history) | derived from the local history: same quarter-hour on d-2, d-3, d-7, d-14, d-21. No d-1 lag |
| Weather | Open-Meteo ICON-D2 single-run API | run as per the table below, 10 grid points per cluster, 13 variables: `temperature_2m`, `dew_point_2m`, `surface_pressure`, `shortwave_radiation`, `direct_radiation`, `diffuse_radiation`, `cloud_cover`, `precipitation`, `snow_depth`, `wind_speed_80m`, `wind_direction_80m`, `wind_speed_120m`, `wind_direction_120m`. Aggregated with population weights (means, daily statistics, spatial quantiles, spreads, heating/cooling degree days) |

### 2.2 Inputs per cutoff

| Cutoff | Run start | ICON-D2 run | Realized load on d-1 up to (end of last quarter-hour) | ENTSO-E load forecast for d | Mode |
|---|---|---|---|---|---|
| 07:00 | 06:40 | 03 UTC | 05:30 | no | direct |
| 08:00 | 07:40 | 03 UTC | 06:30 | no | direct |
| 09:00 | 08:40 | 03 UTC | 07:30 | no | direct |
| 10:00 | 09:40 | 06 UTC | 08:30 | no | direct |
| 11:00 | 10:40 | 06 UTC | 09:30 | yes | correction |
| 12:00 | 11:40 | 06 UTC | 10:30 | yes | correction |

- **Direct mode** predicts realized load. **Correction mode** predicts the error
  of the ENTSO-E load forecast and adds it to that forecast.
- The column "Realized load on d-1 up to" applies to:
  1. the morning features of d-1 (and the comparison window on d-8),
  2. the **training labels**: the model is trained on realized load (direct) or
     ENTSO-E forecast error (correction) only up to this time on d-1. Set
     `target_availability_cutoff_hour/minute` accordingly. It must not be left
     unset, which would train on labels up to 23:45 on d-1.
- Set `require_weather_for_training: true` (with
  `weather_presence_column: weather_weighted_t2m_C`), so training days without
  weather features are dropped instead of being learned from. New code option,
  copy `config/load_forecast.py` and `pipelines/load_forecast.py` from the
  research repo.

### 2.3 Operational data flow

| Step | Action |
|---|---|
| Static | Population weights and cluster assignment are stored once as a fixed snapshot |
| Each run | Re-fetch the last 7-14 days of ENTSO-E realized load, overwrite them, and append the new data; derive lags and morning features from this history |
| 11:00 and 12:00 runs | Fetch the ENTSO-E day-ahead load forecast for d (available from about 10:30) |
| Each run | Fetch the fixed ICON-D2 run for this cutoff from the Open-Meteo single-run API and append it to that run's history |
| Each run | Compute calendar features locally |

### 2.4 Backtest note

For the backtest, ICON-D2 00 and 03 UTC runs are not available for
February-March 2026 (Open-Meteo keeps about six months of single runs). The
07:00-09:00 backtest models therefore use the 06 UTC run and are disclosed as
optimistic in the thesis. Deployment uses the runs in the table above.

## 3. Solar model

### 3.1 Sources

| Source | Access | Available (summer, binding) |
|---|---|---|
| ICON-D2 GRIB | DWD Open Data file server `opendata.dwd.de`: download, aggregate locally, delete GRIB | about 80 min after run start (00 UTC: 03:21, 03 UTC: 06:21, 06 UTC: 09:21 CEST) |
| ICON-D2 cloud cover | Open-Meteo single-run API | about 1.4 h after run start |
| Realized solar per control area (50Hertz, Amprion, TenneT, TransnetBW) and DE-LU total | ENTSO-E Transparency Platform | at most 1 h after the end of each quarter-hour |
| Installed PV capacity with location | Marktstammdatenregister (MaStR) | public, updated daily |

Both weather channels must come from the **same ICON-D2 run**. Each variable
always comes from the same source, in training and in live operation. Mixing
sources (e.g. GRIB history, Open-Meteo for new days) would create a
train/predict mismatch.

### 3.2 Inputs used at every cutoff

| Input | Source | Operational handling |
|---|---|---|
| Radiation and surface weather | DWD ICON-D2 GRIB | `aswdir_s`, `aswdifd_s`, `t_2m`, `td_2m`, `tot_prec`, `h_snow`, `snow_gsp`; 25 clusters per TSO proxy area, capacity-weighted, plus spread measures |
| Cloud cover | Open-Meteo ICON-D2 | `cloud_cover`, `cloud_cover_low`, `cloud_cover_mid`, `cloud_cover_high` at capacity points |
| Installed capacity | MaStR | monthly capacity series per TSO proxy area from a **fixed snapshot** (not updated, see 1.1) |
| Static mappings | local files | cluster assignment, federal-state-to-TSO mapping |
| Solar geometry and physics baseline | computed locally | elevation, azimuth, clear-sky potential, baseline from irradiance, geometry, temperature, and capacity |
| Calendar | computed locally | no fetch |
| Realized generation | ENTSO-E | **not a lagged input.** Used only as training labels (per control area), for the suspicious-observation filter, and for the rolling bias correction (per area and on the DE-LU total) |

### 3.3 Inputs per cutoff

| Cutoff | Run start | ICON-D2 run (GRIB and Open-Meteo) | Realized generation on d-1 up to (end of last quarter-hour) |
|---|---|---|---|
| 07:00 | 06:40 | 03 UTC | 05:30 |
| 08:00 | 07:40 | 03 UTC | 06:30 |
| 09:00 | 08:40 | 03 UTC | 07:30 |
| 10:00 | 09:40 | 06 UTC | 08:30 |
| 11:00 | 10:40 | 06 UTC | 09:30 |
| 12:00 | 11:40 | 06 UTC | 10:30 |

- The realized-generation limit applies to the training labels, the
  suspicious-observation filter, and the rolling bias correction. Set
  `target_availability_cutoff_hour/minute` accordingly. **Exception at 12:00:**
  use 10:00, exactly as in the thesis model (slightly more conservative than the
  rule), so that the deployed 12:00 model is identical to the thesis model.
- **Tight spots at 07:00 and 10:00:** the DWD 03 UTC GRIB is complete at
  06:21 CEST and the 06 UTC GRIB at 09:21 CEST, leaving about 19 minutes for
  download, aggregation, and feature building before the 06:40 and 09:40 run
  starts. Monitor these jobs. If the run is not complete in time, the fallback
  rule in section 1 uses the previous run (00 UTC at 07:00, 03 UTC at 10:00). In
  winter the runs are available one hour earlier.

### 3.4 Operational data flow

| Step | Action |
|---|---|
| Static | MaStR capacity snapshot, cluster assignment, and state-to-TSO mapping stored once |
| Each run | Download the fixed ICON-D2 run from DWD Open Data, aggregate to clusters, build regional features, append them to that run's history, delete GRIB |
| Each run | Fetch cloud cover for the same run from Open-Meteo and append it to that run's history |
| Each run | Re-fetch the last 7-14 days of ENTSO-E realized solar (per control area and DE-LU total), overwrite them, and append the new data |
| Each run | Compute geometry, physics baseline, and calendar locally |

### 3.5 Backtest note

The 12:00 model (RQ1/RQ2) is backtestable exactly as specified: the DWD 06 UTC
archive and the Open-Meteo cloud cache cover the whole period from late
October 2025, and the pipeline trains only on days with weather. The 07:00-09:00
cutoffs lack the 00/03 UTC runs for February-March (see 2.4). Residual
limitations: the backtest uses the current MaStR snapshot, which contains
late-registered units, and ENTSO-E values after later revisions.

## 4. Wind model

Operationally identical in structure to the solar model (section 3): DWD GRIB
and Open-Meteo for the same fixed ICON-D2 run, ENTSO-E realized values, static
MaStR snapshot, local calendar. The differences are listed below.

### 4.1 Sources

| Source | Access | Available (summer, binding) |
|---|---|---|
| ICON-D2 GRIB | DWD Open Data file server: download, aggregate locally, delete GRIB | about 80 min after run start (03:21 / 06:21 / 09:21 CEST) |
| ICON-D2 wind at height and more | Open-Meteo single-run API | about 1.4 h after run start (03:24 / 06:24 / 09:24 CEST) |
| Realized wind onshore and offshore | ENTSO-E Transparency Platform | at most 1 h after the end of each quarter-hour |
| Installed wind capacity with location (onshore, offshore) | MaStR, fixed snapshot | static |

**ICON-D2 only.** Other Open-Meteo providers are excluded. Their 06 UTC runs are
published 3.7-8.1 h after run start (ECMWF 7.7 h, UKMO 8.1 h, GFS 6.5 h, ARPEGE
3.9 h, ICON-EU 3.8 h, HARMONIE 3.7 h, measured 2026-09-27), i.e. after the 12:00
deadline. Their 00 UTC runs would be admissible at 12:00 but are older than the
ICON-D2 06 UTC run. Of these, only GFS and ARPEGE have a usable single-run
history from October 2025 (ECMWF, UKMO, and HARMONIE return no wind at height
via Open-Meteo). A GFS/ARPEGE 00 UTC extension is possible later but is not part
of the thesis model.

### 4.2 Inputs used at every cutoff

| Input | Source | Operational handling |
|---|---|---|
| Surface weather | DWD ICON-D2 GRIB | `t_2m`, `td_2m`, `p`, `u_10m`, `v_10m`, `vmax_10m`, `tot_prec`, `h_snow`, `snow_gsp`; 100 clusters including offshore, grouped into 6 wind regions (offshore, north, west, central, east, south), capacity-weighted. Derived: hub-height wind extrapolated from 10 m to 100 m (power law, exponent 0.14), power-curve and cubic transforms, direction terms |
| Wind at height | Open-Meteo ICON-D2 | `wind_speed_80m`, `wind_direction_80m`, `wind_speed_120m`, `wind_direction_120m`, `wind_speed_180m`, `wind_direction_180m`, `temperature_2m`, `surface_pressure`, `boundary_layer_height`, `cloud_cover` at capacity points (10 per onshore region, 30 offshore). `boundary_layer_height` is always null for ICON-D2; keep requesting it so the feature set stays identical to the thesis model, but exclude it from the null check (rule in 1). The wind variables are required: if they are null, use the previous run |
| Forecast-derived features | computed from the weather forecast | lags, leads, and differences of the forecast features within the forecast day (4 quarter-hours), wind-direction and high-wind regimes. These are shifts of the forecast, not realized data |
| Installed capacity | MaStR | onshore and offshore capacity from the fixed snapshot |
| Static mappings | local files | cluster and region assignment |
| Calendar | computed locally | no fetch |
| Realized generation | ENTSO-E | **not a lagged input.** Used only as training labels (onshore and offshore separately) and for the rolling bias correction (30 days, per quarter-hour) |

### 4.3 Inputs per cutoff

| Cutoff | Run start | ICON-D2 run (GRIB and Open-Meteo) | Realized generation on d-1 up to (end of last quarter-hour) |
|---|---|---|---|
| 07:00 | 06:40 | 03 UTC | 05:30 |
| 08:00 | 07:40 | 03 UTC | 06:30 |
| 09:00 | 08:40 | 03 UTC | 07:30 |
| 10:00 | 09:40 | 06 UTC | 08:30 |
| 11:00 | 10:40 | 06 UTC | 09:30 |
| 12:00 | 11:40 | 06 UTC | 10:30 |

- The realized-generation limit applies to training labels and bias correction
  (`target_availability_cutoff_hour/minute`). **Exception at 12:00:** use 10:00,
  exactly as in the thesis model, so that the deployed 12:00 model is identical
  to the thesis model.
- Same tight spots at 07:00 and 10:00 as for solar (section 3.3).

### 4.4 Differences from solar

| | Solar | Wind |
|---|---|---|
| DWD clusters | 25 per TSO proxy area | 100 including offshore, 6 regions |
| Targets | 4 control areas | onshore and offshore |
| Training window | 90 days | 180 days (minimum 90), so the transition for the early runs takes longer |
| Bias correction | 45 days, per hour | 30 days, per quarter-hour |
| Energy-Arena submission | total | onshore only |

### 4.5 Deployment change

The live wind model still uses the seven-provider ensemble (with fallback to
available runs). Switch it to the ICON-D2-only model so that it matches the
thesis model.

### 4.6 Backtest note

The seven-provider version used in earlier results requested the 06 UTC run of
all providers, which is not admissible before 12:00 (verified for ECMWF: the
cached data matched the 06 UTC run exactly). The ICON-D2-only rerun fixes this.

- Warm-up for the price model: with a minimum of 90 training days the warm-up
  forecasts would start only on 2026-01-24, so the warm-up variant uses
  `min_train_days: 20`, as solar.
- Data gaps: for delivery days 2026-05-14, 07-09, and 07-29 the Open-Meteo 06 UTC
  wind fields were null. The backtest uses the 03 UTC run for them (rule in 1).
  For 2026-02-07 no run is archived any more, so this day is excluded from all
  evaluations involving wind.
- Results: matched RMSE about 3,150 MW vs ENTSO-E about 2,260 MW before the data
  fix (the seven-provider version was 2,707 MW, partly from the late runs).

## 5. Price model

### 5.1 Sources

| Source | Access | Available |
|---|---|---|
| ICON-D2 GRIB | DWD Open Data file server (download, aggregate to 2 clusters, delete GRIB) | about 80 min after run start (03:21 / 06:21 / 09:21 CEST) |
| Day-ahead prices (SDAC) | ENTSO-E Transparency Platform | around 12:45 on d-2 for delivery day d-1, so prices up to and including delivery day d-1 are known at every cutoff |
| Load, solar, and wind forecasts | own component models of the **same cutoff** (sections 2-4) | directly after the component runs |
| Control-reserve auction results | regelleistung.net | gate closure on d-1 at 08:00 (FCR), 09:00 (aFRR), 10:00 (mFRR); publication time not documented (see 5.5) |
| EXAA day-ahead prices for Germany | ENTSO-E Transparency Platform | observed around 11:15 on d-1 (auction closes 10:15) |

### 5.2 Inputs used at every cutoff

| Input | Content |
|---|---|
| Price lags | day-ahead price of the same quarter-hour on d-1, d-2, d-7 |
| Calendar | weekday, holiday, 15-minute market indicator; computed locally |
| Raw weather | DWD ICON-D2 `t_2m`, `u_10m`, `v_10m`, `aswdir_s`, `aswdifd_s`; 2 clusters |
| Self-generated forecasts | load L, renewable generation G = S + W, residual load RL = L - G, residual-load ramps (1 h, 3 h); all from the component models of the same cutoff |
| Training labels | day-ahead prices of past delivery days (complete at every cutoff, no cutoff limit needed) |

The model is fitted separately for each of the 96 quarter-hours on a 70-day
rolling window, with a variance-stabilizing transformation of the target.

### 5.3 Inputs per cutoff

| Cutoff | Run start | ICON-D2 run (raw weather) | Load forecast from | Solar and wind forecasts from | Reserve results | EXAA |
|---|---|---|---|---|---|---|
| 07:00 | 06:40 | 03 UTC | 07:00 load model (direct, 03 UTC, actuals to 05:30) | 07:00 solar and wind (03 UTC, actuals to 05:30) | none | no |
| 08:00 | 07:40 | 03 UTC | 08:00 load model (direct, 03 UTC, actuals to 06:30) | 08:00 solar and wind (03 UTC, actuals to 06:30) | none | no |
| 09:00 | 08:40 | 03 UTC | 09:00 load model (direct, 03 UTC, actuals to 07:30) | 09:00 solar and wind (03 UTC, actuals to 07:30) | FCR | no |
| 10:00 | 09:40 | 06 UTC | 10:00 load model (direct, 06 UTC, actuals to 08:30) | 10:00 solar and wind (06 UTC, actuals to 08:30) | FCR, aFRR | no |
| 11:00 | 10:40 | 06 UTC | 11:00 load model (correction, 06 UTC, actuals to 09:30) | 11:00 solar and wind (06 UTC, actuals to 09:30) | FCR, aFRR, mFRR | no |
| 12:00 | 11:40 | 06 UTC | 12:00 load model (correction, 06 UTC, actuals to 10:30) | 12:00 solar and wind (06 UTC, actuals to 10:30) | FCR, aFRR, mFRR | yes |

Reserve columns: FCR demand, capacity price, surplus; aFRR positive and
negative average and marginal capacity price, allocated, offered,
import/export; mFRR positive and negative average and marginal capacity price,
offered, import/export.

**Component forecasts in training.** The 70 training days must use the
component forecasts **of the same cutoff** for those days (what the 07:00 load
model predicted, not the 12:00 forecast). Store a separate history of component
forecasts per cutoff. A new cutoff needs a 70-day warm-up of its component
forecasts before its price model is consistent.

### 5.3.1 Storing the component forecasts

The price model of each cutoff trains on the component forecasts of the same
cutoff for the past 70 days. These forecasts must therefore be stored and
archived, not only submitted.

| Rule | Detail |
|---|---|
| One history per cutoff and model | e.g. `load_0700`, `solar_0700`, `wind_0700`, ..., `wind_1200`; one row per delivery day and quarter-hour |
| Write before the price run | each component run writes its forecast to its local history immediately (step 2 in 5.4); the price model reads from there |
| Store what was actually used | if a fallback forecast was used (retry failed, cached or filled values), store that forecast together with a fallback flag. The history must reflect what the price model really received, not a later regenerated version |
| Never overwrite past days | a delivery day's stored forecast is final once the cutoff has passed; later reruns go into a separate folder |
| Archive daily | include all per-cutoff component-forecast histories in the daily archive export and commit (currently 14:00, GitHub LFS) and the Synergie backup |
| Warm-up for a new cutoff | the first 70 days can be back-filled by running the component models on the stored per-run weather histories (backtest mode), flagged as back-filled |

### 5.4 Order and runtime within a cutoff job

1. Fetch data: the fixed weather run, ENTSO-E realized values and prices,
   reserve results and EXAA where admissible.
2. Run the load, solar, and wind models (in parallel if possible) and write
   their forecasts to the per-cutoff histories (5.3.1).
3. Run the price model with the forecasts from step 2.
4. Submit, and log any fallback used.

Measured model runtimes per forecast day (backtest, excluding data fetching):
load 6 s, solar 17 s (max 70 s), wind 25 s (max 81 s), price 81 s. The models
take 2-4 min sequentially, well within the 20-minute budget. The binding
constraint is data availability, in particular the DWD 06 UTC GRIB at 10:00.

### 5.5 Reserve timing (measured)

With the run-start rule, reserve results must be available **40 minutes after
gate closure** (FCR by 08:40, aFRR by 09:40, mFRR by 10:40). The publication
time is not documented by regelleistung.net. Measured on 2026-09-28 (delivery
day 2026-09-29) with `scripts/poll_reserve_publication.py`, polling once per
minute: FCR 08:14, aFRR 09:16, mFRR 10:23, i.e. 14-23 minutes after gate closure
and well within the 40-minute budget. **Enable the reserve blocks** as in 5.3
(FCR from 09:00, aFRR from 10:00, mFRR from 11:00). Keep the daily poll on the
VM running to confirm this over more days. If results arrive later than 40 minutes after gate closure, each
reserve block must move one cutoff later.

### 5.6 Backtest note

The 12:00 price model without reserves and EXAA corresponds to P_gen in RQ2.
Price evaluations exclude 2026-02-07 (wind, see 4.6) and 2026-06-19 (weather
unavailable).
The cutoff backtest uses the same structure with the per-cutoff component
forecasts. For 07:00-09:00 the backtest uses the 06 UTC run instead of 00/03
UTC (see 2.4).
