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
| `docs/` | Rerun plan of the final runs and the operational cutoff data specification |
| `Final Paper/` | LaTeX source of the thesis, the submitted PDF, and the figure generators that read `results/` |
| `data/clustering/` | ICON-D2 grid-cluster assignments used for the spatial weather aggregation |
| `tests/` | Unit tests |

Raw and processed input data (ICON-D2 GRIB archives, ENTSO-E downloads,
MaStR, aggregated weather features; about 70 GB) are not included. The
fetch and preprocessing code is in `src/` and `configs/preprocessing/`.

## Running a model

Every model run is one config:

```
pixi run -e forecast da-price-forecast --config <config.yaml>
```

The output folder is the config's `export_dir`. API keys go in a local `.env`
(template: `.env.example`).

## Thesis to repository map

The test period is 1 February to 31 July 2026. All configs suffixed `_clean`
are the leakage-free final reruns reported in the thesis.

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
- `docs/rerun_plan.md` documents the final leakage-free reruns and the checks
  applied to them.
