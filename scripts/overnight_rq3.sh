#!/usr/bin/env bash
# Overnight RQ3 chain: wait for the rerun solar components, verify them, then run the
# RQ3 price stage and the RQ3 SHAP diagnostics. Stops at the first failed step.
#
#   nohup caffeinate -i bash scripts/overnight_rq3.sh > logs/rq3_clean/overnight.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
LOG=logs/rq3_clean
say() { echo "[$(date '+%d.%m %H:%M')] $*"; }

say "waiting for the 5 solar fix runs"
while true; do
  done_n=0
  for t in 0700 0800 0900 1000 1100; do
    grep -q -i "saved" "$LOG/solar_${t}_fix.log" 2>/dev/null && done_n=$((done_n + 1))
    if grep -q -i "traceback" "$LOG/solar_${t}_fix.log" 2>/dev/null; then say "solar_$t failed, stopping"; exit 1; fi
  done
  [ "$done_n" -eq 5 ] && break
  sleep 120
done
say "solar runs finished"

if ! pixi run -e forecast python scripts/check_solar_fix.py; then
  say "solar check FAILED, price stage not started"; exit 1
fi

say "starting RQ3 price stage"
bash scripts/run_rq3_clean.sh prices
if grep -q "FAIL" <(grep "price_" "$LOG/_status.log" | tail -10); then
  say "a price model failed, SHAP not started"; exit 1
fi

say "starting RQ3 SHAP"
mkdir -p results/feature_importance/rq3_clean
for pair in "price_0900 price_1000" "price_1100 price_1200"; do
  for c in $pair; do
    pixi run -e forecast python scripts/shap_stream.py --config configs/rq3_clean/$c.yaml --label $c \
      --output-dir results/feature_importance/rq3_clean --workers 4 > "$LOG/shap_$c.log" 2>&1 &
  done
  wait
done
say "all done"
