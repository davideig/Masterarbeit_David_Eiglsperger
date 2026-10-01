"""Measure when regelleistung.net publishes the day-ahead capacity auction results.

Polls the aggregated-results endpoint once per minute for the next delivery day
and logs, for each product, the first local time at which a non-empty result
file is returned. Run it on the morning of day d-1, e.g. from 07:55 to 11:20. If started after
11:20 (e.g. the evening before), it waits until 07:55 the next morning:

    pixi run -e forecast python scripts/poll_reserve_publication.py

Results are appended to logs/reserve_publication_times.csv.
"""
from __future__ import annotations

import io
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests

URL = "https://www.regelleistung.net/apps/crds/api/v2/tenders/results/aggregated"
PRODUCTS = ["FCR", "aFRR", "mFRR"]
TZ = ZoneInfo("Europe/Berlin")
START_AT = "07:55"
STOP_AT = "11:20"
OUT = Path("logs/reserve_publication_times.csv")


def published(product: str, delivery: date) -> bool:
    params = {"productType": product, "market": "CAPACITY", "exportFormat": "xlsx",
              "deliveryDate": delivery.isoformat()}
    try:
        r = requests.get(URL, params=params, timeout=60)
    except requests.RequestException:
        return False
    if r.status_code != 200 or not r.content.startswith(b"PK"):
        return False
    try:
        return len(pd.read_excel(io.BytesIO(r.content))) > 0
    except Exception:
        return False


def _at(day: date, hhmm: str) -> datetime:
    return datetime.combine(day, datetime.strptime(hhmm, "%H:%M").time(), TZ)


def main() -> None:
    now = datetime.now(TZ)
    poll_day = now.date()
    if now >= _at(poll_day, STOP_AT):
        poll_day += timedelta(days=1)
    start = _at(poll_day, START_AT)
    if datetime.now(TZ) < start:
        print(f"Waiting until {start:%Y-%m-%d %H:%M} local time", flush=True)
        while datetime.now(TZ) < start:
            time.sleep(min(300, max(1, (start - datetime.now(TZ)).total_seconds())))
    delivery = poll_day + timedelta(days=1)
    stop = _at(poll_day, STOP_AT)
    found: dict[str, str] = {}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    print(f"Polling results for delivery day {delivery} until {STOP_AT} local time", flush=True)
    while len(found) < len(PRODUCTS) and datetime.now(TZ) < stop:
        for product in PRODUCTS:
            if product in found:
                continue
            if published(product, delivery):
                t = datetime.now(TZ).strftime("%H:%M:%S")
                found[product] = t
                print(f"{t}  {product} published", flush=True)
                with OUT.open("a") as fh:
                    fh.write(f"{delivery},{product},{t}\n")
        time.sleep(60)
    for product in PRODUCTS:
        if product not in found:
            print(f"{product}: not published before {STOP_AT}", flush=True)


if __name__ == "__main__":
    main()
