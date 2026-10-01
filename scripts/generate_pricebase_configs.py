#!/usr/bin/env python3
"""Generate P_base (RQ2 baseline) sweep configs for the DE-LU price-forecasting thesis.

Staged tuning, all on the DEVELOPMENT window (never touch OOS until the very end):

  stage 1  window    : training-window sweep at a fixed mid cluster count (C=16)
  stage 2  hyper      : hyperparameter grid at C=16 + the window chosen in stage 1
  stage 3  clusters   : cluster-count sweep with window + hyperparameters frozen (the Pareto figure)
  stage 4  oos        : single OOS run of the fully frozen configuration

Each later stage takes the choices frozen in the previous one as CLI flags, so nothing
is selected by peeking at OOS. Configs are written to configs/pricebase_sweep/.

Examples:
  python scripts/generate_pricebase_configs.py window
  python scripts/generate_pricebase_configs.py hyper    --window 56
  python scripts/generate_pricebase_configs.py clusters --window 56 --num-leaves 15 \
                                                         --n-estimators 300 --learning-rate 0.03 --reg-lambda 5
  python scripts/generate_pricebase_configs.py oos      --window 56 --clusters 16 --num-leaves 15 \
                                                         --n-estimators 300 --learning-rate 0.03 --reg-lambda 5
"""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import yaml

# --- fixed experiment axes ---------------------------------------------------
FIXED_MID_CLUSTERS = 16
WINDOWS = [28, 42, 56, 70]            # bounded by weather archive start (2025-10-25)
CLUSTERS = [1, 2, 5, 8, 12, 16, 25, 40, 64, 100]
HYPER_GRID = {                        # short stage-2 grid: the two knobs that matter most
    "num_leaves": [7, 15, 31],        # tree complexity
    "n_estimators": [100],            # 100 for fast dev selection; freeze the final model at 300
    "learning_rate": [0.03],
    "reg_lambda": [5.0, 20.0],        # L2 regularization
}
ICON_DIR_TEMPLATE = "data/processed/icon_aggregated_c{n}_run06"

# --- development vs out-of-sample evaluation windows -------------------------
DEV = {
    "entsoe_start_date": "2025-10-01T00:00:00+02:00",
    "entsoe_end_date": "2026-01-31T23:45:00+01:00",
    "test_start": "2025-12-20T00:00:00+01:00",   # common dev-test window: fair across all train windows
    "test_end": "2026-01-31T23:45:00+01:00",
}
OOS = {
    "entsoe_start_date": "2025-10-01T00:00:00+02:00",
    "entsoe_end_date": "2026-07-31T23:45:00+02:00",
    "test_start": "2026-02-01T00:00:00+01:00",
    "test_end": "2026-07-31T23:45:00+02:00",
}

OUT_DIR = Path("configs/pricebase_sweep")
RESULTS_ROOT = "results/price_forecast_results/pricebase"

# Everything that is held constant across every P_base run.
BASE = {
    "repo_root": ".",
    "target_tz": "Europe/Berlin",
    "post_regime_start": "2025-10-01T00:00:00+02:00",
    "country_code_entsoe": "DE_LU",
    "entsoe_api_key_env": "ENTSOE_API_KEY",
    "dwd_folder_offset_date": "2025-10-26",
    "start_folder_date": "2025-10-25",
    "required_run": "06",
    "use_vst": True,
    "weather_source": "DWD",
    "variant": "fundamental",             # weather-based, no EXAA
    "lars_start_date": "2027-01-01T00:00:00+01:00",
    "price_model_type": "lightgbm",       # boosted trees
    "lgbm_subsample": 1.0,
    "lgbm_colsample_bytree": 0.8,
    "lgbm_min_child_samples": 8,
    "random_state": 42,
    "include_load_forecast_features": True,   # ENTSO-E day-ahead load forecast
    "add_calendar_features": True,            # weekday / holiday features
    "features": {"covariates": []},
}


def write_config(name: str, overrides: dict) -> Path:
    cfg = dict(BASE)
    cfg.update(overrides)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{name}.yaml"
    with path.open("w") as handle:
        yaml.safe_dump(
            {"kind": "lear_operational", "plot": False, "config": cfg},
            handle,
            sort_keys=False,
            default_flow_style=False,
        )
    return path


def _hyper(num_leaves, n_estimators, learning_rate, reg_lambda) -> dict:
    return {
        "lgbm_num_leaves": num_leaves,
        "lgbm_n_estimators": n_estimators,
        "lgbm_learning_rate": learning_rate,
        "lgbm_reg_lambda": reg_lambda,
    }


def stage_window() -> list[Path]:
    """Stage 1: sweep train_days_rolling at C=16, default hyperparameters, dev window."""
    default = _hyper(7, 300, 0.03, 5.0)
    paths = []
    for w in WINDOWS:
        name = f"dev_window_c{FIXED_MID_CLUSTERS}_d{w}"
        paths.append(write_config(name, {
            **DEV,
            **default,
            "icon_dir": ICON_DIR_TEMPLATE.format(n=FIXED_MID_CLUSTERS),
            "n_clusters": FIXED_MID_CLUSTERS,
            "train_days_rolling": w,
            "export_dir": f"{RESULTS_ROOT}/stage1_window/{name}",
        }))
    return paths


def stage_hyper(window: int) -> list[Path]:
    """Stage 2: hyperparameter grid at C=16 + chosen window, dev window."""
    paths = []
    for nl, ne, lr, rl in itertools.product(*HYPER_GRID.values()):
        name = f"dev_hyper_c{FIXED_MID_CLUSTERS}_d{window}_nl{nl}_ne{ne}_lr{lr}_rl{int(rl)}"
        paths.append(write_config(name, {
            **DEV,
            **_hyper(nl, ne, lr, rl),
            "icon_dir": ICON_DIR_TEMPLATE.format(n=FIXED_MID_CLUSTERS),
            "n_clusters": FIXED_MID_CLUSTERS,
            "train_days_rolling": window,
            "export_dir": f"{RESULTS_ROOT}/stage2_hyper/{name}",
        }))
    return paths


def stage_clusters(window: int, hyper: dict) -> list[Path]:
    """Stage 3: cluster-count sweep with window + hyperparameters frozen, dev window."""
    paths = []
    for c in CLUSTERS:
        name = f"dev_clusters_c{c}_d{window}"
        paths.append(write_config(name, {
            **DEV,
            **hyper,
            "icon_dir": ICON_DIR_TEMPLATE.format(n=c),
            "n_clusters": c,
            "train_days_rolling": window,
            "export_dir": f"{RESULTS_ROOT}/stage3_clusters/{name}",
        }))
    return paths


def stage_oos(window: int, clusters: int, hyper: dict) -> list[Path]:
    """Stage 4: single OOS run of the fully frozen configuration."""
    name = f"oos_pbase_c{clusters}_d{window}"
    return [write_config(name, {
        **OOS,
        **hyper,
        "icon_dir": ICON_DIR_TEMPLATE.format(n=clusters),
        "n_clusters": clusters,
        "train_days_rolling": window,
        "export_dir": f"{RESULTS_ROOT}/stage4_oos/{name}",
    })]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=["window", "hyper", "clusters", "oos"])
    p.add_argument("--window", type=int, help="frozen training window (stages hyper/clusters/oos)")
    p.add_argument("--clusters", type=int, help="frozen cluster count (stage oos)")
    p.add_argument("--num-leaves", type=int)
    p.add_argument("--n-estimators", type=int)
    p.add_argument("--learning-rate", type=float)
    p.add_argument("--reg-lambda", type=float)
    args = p.parse_args()

    if args.stage == "window":
        paths = stage_window()
    elif args.stage == "hyper":
        assert args.window, "--window required"
        paths = stage_hyper(args.window)
    else:
        assert args.window and args.num_leaves and args.n_estimators and args.learning_rate and args.reg_lambda, \
            "--window --num-leaves --n-estimators --learning-rate --reg-lambda required"
        hyper = _hyper(args.num_leaves, args.n_estimators, args.learning_rate, args.reg_lambda)
        if args.stage == "clusters":
            paths = stage_clusters(args.window, hyper)
        else:
            assert args.clusters, "--clusters required for oos"
            paths = stage_oos(args.window, args.clusters, hyper)

    print(f"Wrote {len(paths)} config(s) to {OUT_DIR}/:")
    for path in paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()
