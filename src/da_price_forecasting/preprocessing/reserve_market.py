from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Literal

import pandas as pd
import requests

from ..config import ReserveMarketConfig

ReserveProduct = Literal["FCR", "aFRR", "mFRR"]


@dataclass(frozen=True)
class DownloadResult:
    path: Path | None
    status: str
    message: str = ""


@dataclass(frozen=True)
class ProductWindow:
    direction: str
    start_hour: int
    end_hour: int


CAPACITY_COLUMNS: dict[str, tuple[str, list[str]]] = {
    "germany_min_capacity_price_eur_mw_h": ("min_capacity_price_eur_mw_h", ["germany", "min", "capacity", "price"]),
    "germany_average_capacity_price_eur_mw_h": (
        "avg_capacity_price_eur_mw_h",
        ["germany", "average", "capacity", "price"],
    ),
    "germany_marginal_capacity_price_eur_mw_h": (
        "marginal_capacity_price_eur_mw_h",
        ["germany", "marginal", "capacity", "price"],
    ),
    "germany_import_export_mw": ("import_export_mw", ["germany", "import", "export", "mw"]),
    "germany_allocated_volume_mw": ("allocated_mw", ["germany", "allocated", "volume", "mw"]),
    "germany_sum_of_offered_capacity_mw": ("offered_mw", ["germany", "sum", "offered", "capacity", "mw"]),
}

FCR_COLUMNS: dict[str, tuple[str, list[str]]] = {
    "germany_demand_mw": ("germany_demand_mw", ["germany", "demand", "mw"]),
    "germany_settlementcapacity_price_eur_mw": (
        "germany_capacity_price_eur_mw",
        ["germany", "settlement", "capacity", "price"],
    ),
    "crossborder_settlementcapacity_price_eur_mw": (
        "crossborder_capacity_price_eur_mw",
        ["crossborder", "settlement", "capacity", "price"],
    ),
    "germany_deficit_surplus_mw": ("germany_surplus_mw", ["germany", "deficit", "surplus", "mw"]),
}


def _normalise_column(value: object) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def _find_column(columns: list[str], tokens: list[str]) -> str | None:
    normalised = {column: _normalise_column(column) for column in columns}
    for column, clean in normalised.items():
        if all(token in clean for token in tokens):
            return column
    return None


def _product_column(df: pd.DataFrame) -> str:
    for candidate in ("PRODUCT", "PRODUCTNAME", "PRODUCT_NAME", "PRODUCT NAME"):
        if candidate in df.columns:
            return candidate
    column = _find_column(list(df.columns), ["product"])
    if column is None:
        raise ValueError("Reserve-market XLSX is missing a product/product-name column.")
    return column


def _parse_product_window(product_label: object, reserve_type_label: object | None = None) -> ProductWindow | None:
    label = str(product_label)
    reserve_type = str(reserve_type_label) if reserve_type_label is not None else ""
    match = re.search(r"(?:(NEG|POS|NEGPOS)[_\s-]*)?(\d{2})[_:\s-]+(\d{2})", label, flags=re.IGNORECASE)
    if match is None:
        return None

    direction = (match.group(1) or reserve_type or "NEGPOS").lower()
    if "neg" in direction and "pos" not in direction:
        direction = "neg"
    elif "pos" in direction and "neg" not in direction:
        direction = "pos"
    else:
        direction = "negpos"

    start_hour = int(match.group(2))
    end_hour = int(match.group(3))
    if start_hour < 0 or start_hour > 23 or end_hour < 1 or end_hour > 24:
        return None
    if end_hour <= start_hour:
        return None
    return ProductWindow(direction=direction, start_hour=start_hour, end_hour=end_hour)


def _coerce_numeric(value: object) -> float:
    if pd.isna(value):
        return float("nan")
    if isinstance(value, str):
        value = value.strip().replace("\u2212", "-")
        if value in {"", "-", "–", "—", "nan", "NaN", "NA", "N/A"}:
            return float("nan")
        value = value.replace(" ", "").replace("\xa0", "")
        value = value.replace(".", "").replace(",", ".") if "," in value else value
    return float(value)


def _delivery_index(delivery_date: date, target_tz: str) -> pd.DatetimeIndex:
    start = pd.Timestamp(delivery_date).tz_localize(target_tz)
    end = pd.Timestamp(delivery_date + timedelta(days=1)).tz_localize(target_tz)
    return pd.date_range(start=start, end=end, freq="15min", inclusive="left", name="timestamp")


def _assumed_publication_timestamp(
    delivery_date: date,
    product_type: ReserveProduct,
    target_tz: str,
    publication_times: dict[str, str],
) -> pd.Timestamp:
    time_value = publication_times.get(product_type)
    if time_value is None:
        return pd.NaT
    hour, minute = (int(part) for part in time_value.split(":"))
    publication_day = delivery_date - timedelta(days=1)
    return pd.Timestamp(publication_day).tz_localize(target_tz) + pd.Timedelta(hours=hour, minutes=minute)


def build_reserve_capacity_features(
    df: pd.DataFrame,
    *,
    product_type: ReserveProduct,
    delivery_date: date,
    target_tz: str = "Europe/Berlin",
) -> pd.DataFrame:
    """Convert one regelleistung.net capacity result sheet to 15-minute model features."""
    if df.empty:
        return pd.DataFrame(index=_delivery_index(delivery_date, target_tz)).rename_axis("timestamp")

    columns = list(df.columns)
    product_column = _product_column(df)
    reserve_type_column = _find_column(columns, ["type", "reserve"])
    index = _delivery_index(delivery_date, target_tz)
    out = pd.DataFrame(index=index)

    if product_type == "FCR":
        prefix = "reserve_fcr"
        source_columns = {
            output_suffix: _find_column(columns, tokens)
            for _, (output_suffix, tokens) in FCR_COLUMNS.items()
        }
    else:
        product_prefix = product_type.lower()
        source_columns = {
            output_suffix: _find_column(columns, tokens)
            for _, (output_suffix, tokens) in CAPACITY_COLUMNS.items()
        }

    for _, row in df.iterrows():
        reserve_type = row[reserve_type_column] if reserve_type_column is not None else None
        window = _parse_product_window(row[product_column], reserve_type)
        if window is None:
            continue

        hour_float = index.hour + index.minute / 60.0
        mask = (hour_float >= window.start_hour) & (hour_float < window.end_hour)
        if not mask.any():
            continue

        if product_type == "FCR":
            row_prefix = prefix
        else:
            direction = "pos" if window.direction == "pos" else "neg"
            row_prefix = f"reserve_{product_prefix}_{direction}"

        for output_suffix, source_column in source_columns.items():
            if source_column is None:
                continue
            out.loc[mask, f"{row_prefix}_{output_suffix}"] = _coerce_numeric(row[source_column])

    out = out.sort_index()
    out.index.name = "timestamp"
    return out.astype(float)


def _read_capacity_result_xlsx(path: Path) -> pd.DataFrame:
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Workbook contains no default style.*",
                module="openpyxl",
            )
            return pd.read_excel(path, engine="openpyxl")
    except ImportError as exc:
        raise ImportError(
            "Reading regelleistung.net XLSX files requires `openpyxl`. "
            "Run `pixi install` after the dependency update, or install openpyxl in the forecast environment."
        ) from exc


def _reserve_result_url(base_url: str) -> str:
    return f"{base_url.rstrip('/')}/tenders/results/aggregated"


def _raw_result_path(raw_dir: Path, product_type: ReserveProduct, delivery_date: date) -> Path:
    return raw_dir / "capacity" / product_type.lower() / f"{delivery_date.isoformat()}_{product_type}_capacity_results.xlsx"


def _download_capacity_result(
    config: ReserveMarketConfig,
    *,
    product_type: ReserveProduct,
    delivery_date: date,
) -> DownloadResult:
    path = _raw_result_path(config.raw_dir, product_type, delivery_date)
    if path.exists() and not config.force_download:
        return DownloadResult(path=path, status="cached")

    params = {
        "productType": product_type,
        "market": "CAPACITY",
        "exportFormat": "xlsx",
        "deliveryDate": delivery_date.isoformat(),
    }
    try:
        response = requests.get(
            _reserve_result_url(config.base_url),
            params=params,
            timeout=config.request_timeout_seconds,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        return DownloadResult(path=None, status="error", message=str(exc))

    if not response.content.startswith(b"PK"):
        text = response.text.strip().replace("\n", " ")[:240]
        return DownloadResult(path=None, status="missing", message=text or "Response was not an XLSX file.")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(response.content)
    return DownloadResult(path=path, status="downloaded")


def _date_range(start_date: date, end_date: date) -> list[date]:
    days = (end_date - start_date).days
    return [start_date + timedelta(days=offset) for offset in range(days + 1)]


def run_reserve_market(config: ReserveMarketConfig) -> None:
    """Build a cached public reserve-capacity covariate archive."""
    print("\n--- Fetching Reserve-Market Capacity Data ---")
    print(f"[REGELLEISTUNG] Date range: {config.start_date}..{config.end_date}")
    print(f"[REGELLEISTUNG] Products: {', '.join(config.product_types)}")
    print(f"[REGELLEISTUNG] Raw directory: {config.raw_dir}")

    feature_frames: list[pd.DataFrame] = []
    metadata_rows: list[dict[str, object]] = []

    for delivery_date in _date_range(config.start_date, config.end_date):
        day_frames: list[pd.DataFrame] = []
        for product_type in config.product_types:
            publication_timestamp = _assumed_publication_timestamp(
                delivery_date,
                product_type,
                config.target_tz,
                config.publication_times,
            )
            result = _download_capacity_result(config, product_type=product_type, delivery_date=delivery_date)
            metadata_rows.append(
                {
                    "delivery_date": delivery_date.isoformat(),
                    "product_type": product_type,
                    "market": "CAPACITY",
                    "status": result.status,
                    "path": str(result.path) if result.path is not None else "",
                    "message": result.message,
                    "assumed_published_at": publication_timestamp.isoformat()
                    if pd.notna(publication_timestamp)
                    else "",
                    "checked_at": datetime.now(timezone.utc).isoformat(),
                }
            )

            if result.path is None:
                message = f"[REGELLEISTUNG] {delivery_date} {product_type}: {result.status} {result.message}".strip()
                if config.fail_on_missing:
                    raise RuntimeError(message)
                print(message)
                continue

            raw = _read_capacity_result_xlsx(result.path)
            features = build_reserve_capacity_features(
                raw,
                product_type=product_type,
                delivery_date=delivery_date,
                target_tz=config.target_tz,
            )
            if not features.empty:
                day_frames.append(features)

        if day_frames:
            day_features = pd.concat(day_frames, axis=1).sort_index()
            day_features = day_features.loc[:, ~day_features.columns.duplicated()]
            feature_frames.append(day_features)

        if delivery_date.day == 1 or delivery_date == config.start_date or delivery_date == config.end_date:
            print(f"[REGELLEISTUNG] Processed through {delivery_date}")

    if not feature_frames:
        raise RuntimeError("No reserve-market capacity features could be built.")

    features = pd.concat(feature_frames).sort_index()
    features = features.loc[~features.index.duplicated(keep="last")]
    features.index.name = "timestamp"
    config.output_file.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(config.output_file)

    if config.metadata_file is not None:
        config.metadata_file.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(metadata_rows).to_csv(config.metadata_file, index=False)

    print(
        f"Saved reserve-market features -> {config.output_file} "
        f"({len(features):,} rows x {features.shape[1]} columns)"
    )
    if config.metadata_file is not None:
        print(f"Saved reserve-market download metadata -> {config.metadata_file}")
