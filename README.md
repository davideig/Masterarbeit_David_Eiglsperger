# Thesis Hand-In: Working Repository

Working repository of the Master's thesis *Open and Operational Day-Ahead
Electricity Price, Load, Solar, and Wind Forecasting* (KIT, IIP, submitted
30.09.2026). It contains the code, configurations, scripts, logs, and final
results behind every number in the thesis. The deployed, user-facing version of
the pipeline is maintained separately in the release repository.

The repository is reduced to what the final thesis version uses. Exploratory
models, superseded experiment configs, and deployment tooling from the
development phase are not included. The shared pipeline modules keep a few
switchable options that the final configs leave disabled (for example
alternative load model types or ENTSO-E error features).

## Layout

| Path | Content |
|---|---|
| `src/da_price_forecasting/` | Pipeline code: data access, preprocessing, features, load/solar/wind/price models, evaluation |
| `configs/` | Configs of the final thesis runs (`final/`, `pricebase_sweep/`, `rq3_clean/`, `rq3_cutoff_grid/`) and the preprocessing that built their inputs (`preprocessing/`) |
| `scripts/` | Experiment drivers and evaluation scripts used for the thesis tables |
| `results/` | Outputs of the final thesis runs only (forecast CSVs, metrics, SHAP exports) |
| `logs/` | Run logs of the final reruns and the measured reserve-market publication times |
| `Final Paper/` | LaTeX source of the thesis, the submitted PDF, and the figure generators that read `results/` |
| `data/clustering/` | ICON-D2 grid-cluster assignments used for the spatial weather aggregation |
| `tests/` | Unit tests |

The model input files (aggregated weather features, ENTSO-E series,
reserve-market features; 1.1 GB) are provided as a separate download, see
below. The raw sources they were built from (ICON-D2 GRIB archives, the MaStR
export, ENTSO-E downloads) are not included. The code that builds the input
files from them is in `src/` and `configs/preprocessing/`.

## Rerunning the models

1. Download `thesis_model_inputs.zip` from the release
   `thesis-submission-2026-09-30` on the Releases page of this repository and
   unpack it in the repository root. It creates `data/processed/`.
2. Install [pixi](https://pixi.sh). The environment is created on the first run.
3. Run a model. Every model run is one config:

   ```
   pixi run -e forecast da-price-forecast --config <config.yaml>
   ```

   The output folder is the config's `export_dir`, so a rerun overwrites the
   stored result of that config.

No API keys are needed, because all inputs are read from the unpacked files.

Order: the price models read the load, solar, and wind forecasts from
`results/`, so the component models come first. For RQ3,
`bash scripts/run_rq3_clean.sh components` runs the component models and
`bash scripts/run_rq3_clean.sh prices` the price models. The script skips every
model whose `forecast.csv` already exists, so delete the `rq3_clean/` folders
under `results/` first to rerun them. A load, solar, or wind
run over the test period takes minutes to about one hour, a price model one to
two hours.

Without the input files, the following still run: both evaluation configs
(`configs/final/benchmarks/evaluation_*.yaml`), `scripts/eval_wind_rq1.py`,
`scripts/rq3_clean_eval.py`, and the unit tests (`pixi run test`). They read
only the stored results.

### Reproducibility

Reruns with the input files reproduce the stored forecasts exactly. This was
checked on slices of the test period for every model type (load in both modes,
solar, wind, the ENTSO-E benchmark, P_gen, the RQ3 component and price models,
and the EXAA-only benchmark).

The one exception is P_base. The thesis run of P_base predates a later change in
the price-model code, so a rerun gives slightly different forecasts with the
same accuracy:

| | Thesis | Rerun |
|---|---|---|
| P_base RMSE (EUR/MWh) | 35.50 | 35.49 |
| P_base MAE (EUR/MWh) | 20.82 | 20.80 |
| P_gen relative to P_base | -11.2% | -11.1% |
| P_gen without raw weather relative to P_base | -10.5% | -10.4% |

All Giacomini-White p-values of Table 6 remain below 0.001. The thesis forecast
is kept in `results/price_forecast_results/pricebase/stage4_oos/oos_pbase_c2_d70/`
and the rerun in `oos_pbase_c2_d70_rerun/` next to it.

## Thesis to repository map

The test period is 1 February to 31 July 2026. All configs suffixed `_clean`
are the final runs reported in the thesis.

### RQ1: OBTF forecasts versus ENTSO-E (Table 5)

| Item | Config | Results |
|---|---|---|
| OBTF load, correction mode | `configs/final/load/load_forecast_hybrid_entsoe_residual_..._paper_febjul_..._clean.yaml` | `results/load_forecast_results/hybrid_entsoe_residual_..._paper_febjul_..._clean/` |
| OBTF load, direct mode (17.3% comparison) | `configs/final/load/load_forecast_direct_actual_..._morning0845_..._clean.yaml` | `results/load_forecast_results/direct_actual_..._morning0845_..._clean/` |
| Load evaluation incl. ENTSO-E | `configs/final/benchmarks/evaluation_load_direct_vs_entsoe_residual_paper_febjul_clean.yaml` | `results/evaluation/load_direct_vs_entsoe_residual_paper_febjul_clean/` |
| OBTF solar | `configs/final/renewable/renewable_generation_dwd_icon_mastr_solar_tso_c25_..._cloud_geometry_..._paper_febjul.yaml` | `results/renewable_generation_results/dwd_icon_mastr_solar_tso_c25_..._cloud_geometry_..._paper_febjul/` |
| OBTF wind (ICON-D2 only) | `configs/final/renewable/renewable_generation_hybrid_dwd_mastr_wind_c100_icond2only_..._paper_febjul.yaml` | `results/renewable_generation_results/hybrid_dwd_mastr_wind_c100_icond2only_..._paper_febjul/` |
| ENTSO-E solar/wind benchmark | `configs/final/benchmarks/entsoe_renewable_forecast_benchmark_paper_febjul.yaml` | `results/renewable_generation_results/entsoe_renewable_forecast_benchmark_paper_febjul/` |
| Wind row of Table 5 | `scripts/eval_wind_rq1.py` | printed to console |

### RQ2: OBTF forecasts in the price model (Table 6, Figures 9 and A2, Table A6)

| Item | Config | Results |
|---|---|---|
| Load input (price warm-up window) | `configs/final/load/price_inputs/..._price_warmup_d70_nov23febjul_..._clean.yaml` | `results/load_forecast_results/..._price_warmup_d70_nov23febjul_..._clean/` |
| Solar input | `configs/final/renewable/price_inputs/renewable_generation_dwd_icon_mastr_solar_..._price_warmup_d70_nov23febjul.yaml` | `results/renewable_generation_results/dwd_icon_mastr_solar_..._price_warmup_d70_nov23febjul/` |
| Wind input | `configs/final/renewable/renewable_generation_hybrid_dwd_mastr_wind_c100_icond2only_..._min20_price_warmup_d70_nov23febjul.yaml` | `results/renewable_generation_results/hybrid_..._icond2only_..._min20_price_warmup_d70_nov23febjul/` |
| P_base | `configs/pricebase_sweep/oos_pbase_c2_d70.yaml` | `results/price_forecast_results/pricebase/stage4_oos/oos_pbase_c2_d70/` |
| P_gen | `configs/pricebase_sweep/oos_pgen_c2_d70_clean.yaml` | `results/price_forecast_results/pricebase/stage4_oos/oos_pgen_c2_d70_clean/` |
| P_gen without raw weather | `configs/pricebase_sweep/oos_pgen_nowea_c2_d70_clean.yaml` | `results/price_forecast_results/pricebase/stage4_oos/oos_pgen_nowea_c2_d70_clean/` |
| Naive benchmarks (realized price of the same local quarter-hour on day d-1 and d-7) | | `results/price_forecast_results/pricebase/naive/` |
| Table 6 evaluation | `configs/final/benchmarks/evaluation_rq2_clean.yaml` | `results/evaluation/rq2_clean/` |
| SHAP (Figures 9, A2, Table A6) | `scripts/shap_stream.py`, `scripts/shap_subgroups.py`, `scripts/shap_beeswarm_export.py` | `results/feature_importance/rq2_clean/` |

### Appendix A3.1: base price model selection (Tables A3, A4, Figure A1)

Development-period sweeps generated by `scripts/generate_pricebase_configs.py`:
`results/price_forecast_results/pricebase/stage1_window/` (training window),
`stage2_hyper/` (hyperparameters), and `stage3_clusters/` (weather clusters).

### RQ3: cutoff grid 07:00 to 12:00 (Table 7, Figures 8 and 10, Tables A5 and A7)

| Item | Where |
|---|---|
| Configs (load, solar, wind 07:00-11:00, price 07:00-12:00 with and without reserve) | `configs/rq3_clean/` |
| Run drivers | `scripts/run_rq3_clean.sh`, `scripts/overnight_rq3.sh`, solar gate check `scripts/check_solar_fix.py` |
| Component results | `results/load_forecast_results/rq3_clean/`, `results/renewable_generation_results/rq3_clean/` |
| Price results (incl. reserve ablation) | `results/price_forecast_results/rq3_clean/` |
| 12:00 components | identical to the RQ2 inputs above |
| EXAA-only benchmark | `results/price_forecast_results/rq3_cutoff_grid/price_1200_exaa_only_c2_d70_lsdf/` |
| Table 7, Holm correction, Table A5 | `scripts/rq3_clean_eval.py` (printed to console) |
| SHAP (Figure 10, Table A7) | `results/feature_importance/rq3_clean/` |
| Run logs | `logs/rq3_clean/`, `logs/rerun_clean/` |

### Figures

The figure generators in `Final Paper/figures/generate_*.py` read the result
folders above. Process diagrams are TikZ files in the same folder.

## Known data handling

- Delivery day 7 February 2026 is excluded where the OBTF wind forecast is
  involved (no archived ICON-D2 run with wind fields). 19 June 2026 is
  excluded for prices.
- Three Open-Meteo ICON-D2 06 UTC runs without wind fields (13 May, 8 July,
  28 July 2026) were replaced by the 03 UTC run of the same day.
- Information rules enforced in the final runs (configs suffixed `_clean`,
  `icond2only`, and `configs/rq3_clean/`):
  - Weather comes from ICON-D2 only. The 06 UTC runs of other weather
    providers are published after 12:00 and are not used.
  - The load, solar, and wind models are trained only on realized values that
    were already published at the forecast creation time
    (`target_availability_cutoff_hour` and `_minute`). The load model is
    trained only on days with weather inputs (`require_weather_for_training`).
  - The RQ3 cutoff models use the same model configurations as RQ1 and RQ2 and
    realized data up to the last quarter-hour published at the start of each run.

## Data sources

The result files and `data/clustering/` contain values derived from the
following sources. Reuse of these values is subject to the terms of the
respective source.

| Source | Used for | Terms |
|---|---|---|
| ENTSO-E Transparency Platform (transparency.entsoe.eu) | Day-ahead prices (incl. EXAA prices for Germany), realized load and generation, ENTSO-E load, solar, and wind forecasts | ENTSO-E Transparency Platform terms and conditions |
| Deutscher Wetterdienst, DWD Open Data (opendata.dwd.de) | ICON-D2 weather forecasts | Datenbasis: Deutscher Wetterdienst, DWD terms of use |
| Open-Meteo (open-meteo.com) | ICON-D2 single-run weather and hub-height wind fields | CC BY 4.0 |
| Bundesnetzagentur, Marktstammdatenregister | Installed solar and wind capacity, capacity-weighted clusters | Datenlizenz Deutschland – Namensnennung – Version 2.0 (dl-de/by-2-0) |
| Eurostat GISCO, population grid 2021 | Population weights of the load weather clusters | © European Union, Eurostat |
| Regelleistung.net | Control-reserve auction results (FCR, aFRR, mFRR) | Regelleistung.net terms of use |
