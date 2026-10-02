#!/usr/bin/env bash
# Run the harmonized RQ3 grid (configs/rq3_clean).
#
#   bash scripts/run_rq3_clean.sh components   # 15 component models (07:00-11:00)
#   bash scripts/run_rq3_clean.sh prices       # 10 price models, after components
#                                              # AND the RQ1/RQ2 12:00 warm-ups are done
#
# Each model writes its own log to logs/rq3_clean/<config>.log; a summary line per
# model goes to logs/rq3_clean/_status.log. Existing results are skipped.
set -u
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
LOG=logs/rq3_clean
mkdir -p "$LOG"
C=configs/rq3_clean

run() {
  local cfg="$C/$1.yaml" export
  export=$(grep -m1 "export_dir:" "$cfg" | awk '{print $2}')
  if [ -f "$export/forecast.csv" ]; then
    echo "[$(date +%T)] SKIP  $1 (already done)" | tee -a "$LOG/_status.log"; return 0
  fi
  echo "[$(date +%T)] START $1" | tee -a "$LOG/_status.log"
  if pixi run -e forecast da-price-forecast --config "$cfg" > "$LOG/$1.log" 2>&1; then
    echo "[$(date +%T)] OK    $1" | tee -a "$LOG/_status.log"
  else
    echo "[$(date +%T)] FAIL  $1 (see $LOG/$1.log)" | tee -a "$LOG/_status.log"
  fi
}
stream() { for c in "$@"; do run "$c"; done; }

need() {  # abort if a prerequisite forecast is missing
  for d in "$@"; do
    [ -f "$d/forecast.csv" ] || { echo "Missing prerequisite: $d"; exit 1; }
  done
}

case "${1:-}" in
  components)
    # wind is the slowest model, so it is split over three streams
    stream load_0700 load_0800 load_0900 load_1000 load_1100 \
           solar_0700 solar_0800 solar_0900 solar_1000 solar_1100 &
    stream wind_0700 wind_0800 &
    stream wind_0900 wind_1000 &
    stream wind_1100 &
    wait
    echo "[$(date +%T)] components finished" | tee -a "$LOG/_status.log"
    ;;
  prices)
    need results/load_forecast_results/hybrid_entsoe_residual_open_meteo_p10_morning1015_daily_weather_lightgbm_price_warmup_d70_nov23febjul_tw224_f180_pop_weighted_quantiles_clean \
         results/renewable_generation_results/hybrid_dwd_mastr_wind_c100_icond2only_run06_onoff_split_wind_hub_p80_common_hgb_wind_struct_minleaf60_maxfeat08_bias30_mtu_s08_d180_cutoff1000_min20_price_warmup_d70_nov23febjul
    for t in 0700 0800 0900 1000 1100; do
      for k in load solar wind; do
        d=$(grep -m1 "export_dir:" "$C/${k}_$t.yaml" | awk '{print $2}'); need "$d"
      done
    done
    stream price_0700 price_0800 price_0900 price_0900_noreserve &
    stream price_1000 price_1000_noreserve price_1100 &
    stream price_1100_noreserve price_1200 price_1200_noreserve &
    wait
    echo "[$(date +%T)] prices finished" | tee -a "$LOG/_status.log"
    ;;
  *)
    echo "usage: bash scripts/run_rq3_clean.sh components|prices"; exit 2 ;;
esac
