from __future__ import annotations

import os
import re
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from urllib.parse import urljoin

import requests

from ..config import IconAggregationConfig


DEFAULT_DWD_ICON_BASE_URL = "https://opendata.dwd.de/weather/nwp/icon-d2/grib/"
DEFAULT_DWD_ICON_VARIABLES = [
    "t_2m",
    "td_2m",
    "p",
    "u_10m",
    "v_10m",
    "vmax_10m",
    "aswdir_s",
    "aswdifd_s",
    "tot_prec",
    "h_snow",
    "snow_gsp",
]
REGULAR_LAT_LON_SINGLE_LEVEL = "icon-d2_germany_regular-lat-lon_single-level_"
REGULAR_LAT_LON_MODEL_LEVEL = "icon-d2_germany_regular-lat-lon_model-level_"


def dwd_issue_day_for_forecast(forecast_date: date, folder_offset_date: date) -> date:
    """Return the DWD issue day that provides weather for a target forecast day."""
    if forecast_date > folder_offset_date:
        return forecast_date - timedelta(days=1)
    return forecast_date


def processed_weather_folder(icon_dir: Path, issue_day: date, run_hour: str) -> Path:
    return icon_dir / f"dwd_icon_daily_{issue_day:%Y%m%d}_{run_hour}"


def processed_weather_folder_is_ready(path: Path) -> bool:
    return path.is_dir() and any(path.glob("*.csv"))


def raw_weather_folder(raw_base_dir: Path, issue_day: date, run_hour: str) -> Path:
    return raw_base_dir / f"dwd_icon_daily_{issue_day:%Y%m%d}" / "icon-d2" / run_hour


def _date_range(start: date, end: date) -> list[date]:
    if start > end:
        return []
    return [start + timedelta(days=offset) for offset in range((end - start).days + 1)]


def _processed_issue_days(icon_dir: Path, run_hour: str) -> list[date]:
    if not icon_dir.exists():
        return []

    pattern = re.compile(rf"^dwd_icon_daily_(\d{{8}})_{re.escape(run_hour)}$")
    issue_days: list[date] = []
    for folder in icon_dir.iterdir():
        if not folder.is_dir() or not processed_weather_folder_is_ready(folder):
            continue
        match = pattern.match(folder.name)
        if match is None:
            continue
        try:
            issue_days.append(datetime.strptime(match.group(1), "%Y%m%d").date())
        except ValueError:
            continue
    return sorted(issue_days)


def missing_dwd_issue_days_to_process(
    *,
    icon_dir: Path,
    run_hour: str,
    issue_days: list[date],
    catch_up_missing_days: bool = True,
    force: bool = False,
) -> list[date]:
    """Return issue days that need raw download and aggregation."""
    issue_days = sorted(set(issue_days))
    if not issue_days:
        return []

    if not catch_up_missing_days:
        return [
            day
            for day in issue_days
            if force or not processed_weather_folder_is_ready(processed_weather_folder(icon_dir, day, run_hour))
        ]

    target_issue_day = max(issue_days)
    processed_days = [day for day in _processed_issue_days(icon_dir, run_hour) if day <= target_issue_day]
    if processed_days:
        start_issue_day = min(max(processed_days) + timedelta(days=1), target_issue_day)
    else:
        start_issue_day = min(issue_days)

    return [
        day
        for day in _date_range(start_issue_day, target_issue_day)
        if force or not processed_weather_folder_is_ready(processed_weather_folder(icon_dir, day, run_hour))
    ]


def required_dwd_issue_days(
    *,
    forecast_start: date,
    forecast_end: date,
    folder_offset_date: date,
) -> list[date]:
    return [
        dwd_issue_day_for_forecast(forecast_date, folder_offset_date)
        for forecast_date in _date_range(forecast_start, forecast_end)
    ]


def _remote_links(url: str, timeout_seconds: int) -> list[str]:
    response = requests.get(url, timeout=timeout_seconds)
    response.raise_for_status()
    return re.findall(r'href=["\']([^"\']+)["\']', response.text)


def _matching_remote_grib_files(
    *,
    variable_url: str,
    issue_day: date,
    run_hour: str,
    timeout_seconds: int,
    model_levels: list[int] | None = None,
) -> list[str]:
    init_token = f"_{issue_day:%Y%m%d}{run_hour}_"
    links = _remote_links(variable_url, timeout_seconds=timeout_seconds)
    allowed_model_levels = {str(level) for level in model_levels or []}
    pattern = re.compile(r"_(\d{10})_(\d{3})_(\d+)_([A-Za-z0-9]+)\.grib2\.bz2$")
    return sorted(
        link
        for link in links
        if link.endswith(".grib2.bz2")
        and init_token in link
        and (
            REGULAR_LAT_LON_SINGLE_LEVEL in link
            or (
                REGULAR_LAT_LON_MODEL_LEVEL in link
                and (not allowed_model_levels or ((match := pattern.search(link)) is not None and match.group(3) in allowed_model_levels))
            )
        )
    )


def _download_file(url: str, target: Path, timeout_seconds: int, force: bool) -> None:
    if target.exists() and target.stat().st_size > 0 and not force:
        print(f"[DWD] Reusing {target.name}")
        return

    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"[DWD] Downloading {url}")
    with requests.get(url, stream=True, timeout=timeout_seconds) as response:
        response.raise_for_status()
        with open(target, "wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)


def download_dwd_icon_d2_run(
    *,
    issue_day: date,
    run_hour: str,
    raw_base_dir: Path,
    variables: list[str] | None = None,
    model_levels: list[int] | None = None,
    base_url: str = DEFAULT_DWD_ICON_BASE_URL,
    timeout_seconds: int = 60,
    request_pause_seconds: float = 0.0,
    force: bool = False,
) -> Path:
    """Download one live DWD ICON-D2 run into the daily raw folder layout."""
    variables = variables or DEFAULT_DWD_ICON_VARIABLES
    raw_run_dir = raw_weather_folder(raw_base_dir, issue_day, run_hour)
    run_url = urljoin(base_url.rstrip("/") + "/", f"{run_hour}/")
    total_files = 0

    for variable in variables:
        variable_url = urljoin(run_url, f"{variable}/")
        files = _matching_remote_grib_files(
            variable_url=variable_url,
            issue_day=issue_day,
            run_hour=run_hour,
            timeout_seconds=timeout_seconds,
            model_levels=model_levels,
        )
        if not files:
            print(f"[DWD] No matching files for {issue_day:%Y%m%d} run {run_hour} variable {variable}.")
            continue

        variable_dir = raw_run_dir / variable
        for filename in files:
            _download_file(
                urljoin(variable_url, filename),
                variable_dir / filename,
                timeout_seconds=timeout_seconds,
                force=force,
            )
            total_files += 1

        if request_pause_seconds > 0:
            time.sleep(request_pause_seconds)

    if total_files == 0:
        raise FileNotFoundError(
            "No DWD ICON-D2 files were downloaded. "
            f"The live DWD open-data endpoint may not expose issue day {issue_day:%Y-%m-%d} run {run_hour} anymore."
        )

    return raw_run_dir


def _infer_n_clusters(icon_dir: Path, explicit_n_clusters: int | None) -> int:
    if explicit_n_clusters is not None:
        return explicit_n_clusters
    match = re.search(r"c(\d+)", icon_dir.name)
    if match:
        return int(match.group(1))
    return 25


def ensure_dwd_icon_weather(
    *,
    repo_root: Path,
    icon_dir: Path,
    forecast_start: date,
    forecast_end: date,
    run_hour: str,
    folder_offset_date: date,
    raw_base_dir: Path,
    shapefile_path: Path,
    n_clusters: int | None = None,
    buffer_km: int = 50,
    cluster_source: str = "grid",
    cluster_output_file: Path | None = None,
    capacity_file: Path | None = None,
    capacity_weighted_aggregation: bool = True,
    variables: list[str] | None = None,
    model_levels: list[int] | None = None,
    base_url: str = DEFAULT_DWD_ICON_BASE_URL,
    timeout_seconds: int = 60,
    request_pause_seconds: float = 0.0,
    catch_up_missing_days: bool = True,
    force: bool = False,
) -> list[date]:
    """Download and aggregate missing DWD ICON-D2 issue days for a forecast window."""
    issue_days = required_dwd_issue_days(
        forecast_start=forecast_start,
        forecast_end=forecast_end,
        folder_offset_date=folder_offset_date,
    )
    days_to_process = missing_dwd_issue_days_to_process(
        icon_dir=icon_dir,
        run_hour=run_hour,
        issue_days=issue_days,
        catch_up_missing_days=catch_up_missing_days,
        force=force,
    )
    if not days_to_process:
        print(f"[DWD] Aggregated ICON weather already available through {max(issue_days).isoformat()}.")
        return []

    print("[DWD] Missing ICON issue days: " + ", ".join(day.isoformat() for day in days_to_process))
    for issue_day in days_to_process:
        download_dwd_icon_d2_run(
            issue_day=issue_day,
            run_hour=run_hour,
            raw_base_dir=raw_base_dir,
            variables=variables,
            model_levels=model_levels,
            base_url=base_url,
            timeout_seconds=timeout_seconds,
            request_pause_seconds=request_pause_seconds,
            force=force,
        )

    from .icon_d2_aggregation import run_aggregation

    for issue_day in days_to_process:
        aggregation_config = IconAggregationConfig(
            repo_root=repo_root,
            lsdf_base=raw_base_dir,
            output_parent=icon_dir,
            shapefile_path=shapefile_path,
            n_clusters=_infer_n_clusters(icon_dir, n_clusters),
            buffer_km=buffer_km,
            cluster_source=cluster_source,
            cluster_output_file=cluster_output_file,
            capacity_file=capacity_file or IconAggregationConfig.model_fields["capacity_file"].default,
            capacity_weighted_aggregation=capacity_weighted_aggregation,
            plot_clusters=False,
            skip_existing_output=not force,
            only_day=issue_day.strftime("%Y%m%d"),
            only_run_hour=run_hour,
            start_date=issue_day,
            variables=variables or DEFAULT_DWD_ICON_VARIABLES,
            model_levels=model_levels or [],
        )
        run_aggregation(aggregation_config)

    missing_after = [
        day
        for day in days_to_process
        if not processed_weather_folder_is_ready(processed_weather_folder(icon_dir, day, run_hour))
    ]
    if missing_after:
        raise FileNotFoundError(
            "DWD ICON aggregation completed but processed folders are still missing: "
            + ", ".join(day.isoformat() for day in missing_after)
        )

    return days_to_process
