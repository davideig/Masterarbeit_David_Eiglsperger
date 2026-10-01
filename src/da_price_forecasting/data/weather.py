from __future__ import annotations

import os
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests


class OpenMeteoModelRunUnavailable(RuntimeError):
    """Raised when Open-Meteo reports a missing historical model run."""


def _is_open_meteo_model_run_unavailable(message: object) -> bool:
    text = str(message)
    text_lower = text.lower()
    return "modelrununavailable" in text or ("run" in text_lower and "unavailable" in text_lower)


def _open_meteo_error_reason(response: requests.Response) -> str | None:
    try:
        payload = response.json()
    except ValueError:
        text = response.text.strip()
        return text or None

    if isinstance(payload, dict):
        reason = payload.get("reason")
        if reason:
            return str(reason)
        if payload.get("error"):
            return str(payload)
    return None


def _raise_open_meteo_http_error(response: requests.Response) -> None:
    reason = _open_meteo_error_reason(response)
    if reason and _is_open_meteo_model_run_unavailable(reason):
        raise OpenMeteoModelRunUnavailable(reason[:500])
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        if reason:
            raise requests.HTTPError(f"{exc}; Open-Meteo reason: {reason[:500]}", response=response) from exc
        raise


def _format_open_meteo_run(timestamp: pd.Timestamp) -> str:
    return timestamp.strftime("%Y-%m-%dT%H:%M")


def _forecast_days_for_target(
    *,
    run: str,
    target_day: pd.Timestamp | None,
    default_forecast_days: int | None,
) -> int | None:
    if target_day is None or default_forecast_days is None:
        return default_forecast_days
    run_timestamp = pd.Timestamp(run)
    required_days = (target_day.date() - run_timestamp.date()).days + 1
    return max(default_forecast_days, required_days)


def _open_meteo_run_candidates(
    *,
    run: str | None,
    target_day: pd.Timestamp | None,
    forecast_days: int | None,
    fallback_previous_runs: bool,
    fallback_step_hours: int,
    fallback_max_lookback_hours: int,
) -> list[tuple[str | None, int | None]]:
    if run is None:
        return [(None, forecast_days)]

    candidates = [(run, _forecast_days_for_target(run=run, target_day=target_day, default_forecast_days=forecast_days))]
    if not fallback_previous_runs:
        return candidates
    if fallback_step_hours <= 0:
        raise ValueError("fallback_step_hours must be positive.")
    if fallback_max_lookback_hours < fallback_step_hours:
        return candidates

    requested = pd.Timestamp(run)
    seen = {run}
    for lookback_hours in range(fallback_step_hours, fallback_max_lookback_hours + 1, fallback_step_hours):
        fallback_run = _format_open_meteo_run(requested - pd.Timedelta(hours=lookback_hours))
        if fallback_run in seen:
            continue
        seen.add(fallback_run)
        candidates.append(
            (
                fallback_run,
                _forecast_days_for_target(
                    run=fallback_run,
                    target_day=target_day,
                    default_forecast_days=forecast_days,
                ),
            )
        )
    return candidates


def _fetch_open_meteo_batch_with_run_fallback(
    *,
    base_url: str,
    latitude: list[float],
    longitude: list[float],
    hourly_variables: list[str],
    model: str | None,
    cell_selection: str,
    timeout_seconds: int,
    start_date: date | None = None,
    end_date: date | None = None,
    target_day: pd.Timestamp | None = None,
    run: str | None = None,
    forecast_days: int | None = None,
    retry_attempts: int = 5,
    retry_backoff_seconds: float = 30.0,
    api_key_env: str | None = None,
    fallback_previous_runs: bool = False,
    fallback_step_hours: int = 2,
    fallback_max_lookback_hours: int = 24,
) -> tuple[list[dict], str | None, int | None]:
    last_error: OpenMeteoModelRunUnavailable | None = None
    for candidate_run, candidate_forecast_days in _open_meteo_run_candidates(
        run=run,
        target_day=target_day,
        forecast_days=forecast_days,
        fallback_previous_runs=fallback_previous_runs,
        fallback_step_hours=fallback_step_hours,
        fallback_max_lookback_hours=fallback_max_lookback_hours,
    ):
        try:
            items = _fetch_open_meteo_batch(
                base_url=base_url,
                latitude=latitude,
                longitude=longitude,
                start_date=start_date,
                end_date=end_date,
                run=candidate_run,
                forecast_days=candidate_forecast_days,
                hourly_variables=hourly_variables,
                model=model,
                cell_selection=cell_selection,
                timeout_seconds=timeout_seconds,
                retry_attempts=retry_attempts,
                retry_backoff_seconds=retry_backoff_seconds,
                api_key_env=api_key_env,
            )
        except OpenMeteoModelRunUnavailable as exc:
            last_error = exc
            continue

        if candidate_run != run:
            day_label = target_day.date() if target_day is not None else start_date
            print(
                "[OPEN-METEO] Using fallback model run "
                f"{candidate_run} instead of {run} for day {day_label}."
            )
        return items, candidate_run, candidate_forecast_days

    if last_error is not None:
        raise last_error
    raise RuntimeError("Open-Meteo run fallback exhausted without a request attempt.")


def _read_dwd_variable_name(path: Path) -> str:
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if not line.startswith("#"):
                break
            if line.startswith("# Variable:"):
                return line.split(":", 1)[1].strip()
    return path.name.replace(".csv", "").split("_")[0]


def _canonical_dwd_variable_name(raw_name: str) -> str:
    key = raw_name.strip().lower().replace("-", "_")
    compact = key.replace("_", "")

    if compact in {"aswdirs", "aswdir"}:
        return "ASWDIR"
    if compact in {"aswdifds", "aswdifd"}:
        return "ASWDIFD"
    if compact in {"t2m", "2t"}:
        return "t2m"
    if compact in {"td2m", "d2m", "2d"}:
        return "td2m"
    if compact in {"u10", "10u"}:
        return "u10"
    if compact in {"v10", "10v"}:
        return "v10"
    if compact in {"vmax10m", "vmax10", "10fg", "fg10"}:
        return "vmax10m"
    if compact in {"p", "sp", "pres", "pressure"}:
        return "sp"
    if compact in {"tp", "totprec", "totalprecipitation"}:
        return "tp"
    if compact in {"sde", "hsnow", "snowdepth"}:
        return "sde"
    if compact in {"snowgsp", "snow", "snowfall", "lsfwe"}:
        return "snow_gsp"
    if compact in {"clct", "tcc", "totalcloudcover", "cloudcover"}:
        return "tcc"
    if compact in {"clcl", "lcc", "lowcloudcover"}:
        return "lcc"
    if compact in {"clcm", "mcc", "mediumcloudcover", "midcloudcover"}:
        return "mcc"
    if compact in {"clch", "hcc", "highcloudcover"}:
        return "hcc"
    return key


def load_era5(
    dirs: list[Path],
    target_tz: str = "Europe/Berlin",
) -> pd.DataFrame:
    """Load clustered ERA5 CSV files from multiple yearly directories."""
    yearly_dfs = []

    for base_dir in dirs:
        csv_files = sorted(filename for filename in os.listdir(base_dir) if filename.endswith(".csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in directory: {base_dir}")

        variable_dfs = []
        for filename in csv_files:
            var_name = filename.split("_")[0]
            df_var = pd.read_csv(
                base_dir / filename,
                comment="#",
                parse_dates=["timestamp"],
            )
            df_var = df_var.set_index("timestamp")
            df_var = df_var.rename(columns={col: f"{var_name}_{col}" for col in df_var.columns})
            variable_dfs.append(df_var)

        df_year = variable_dfs[0].copy()
        for df_next in variable_dfs[1:]:
            df_year = df_year.join(df_next, how="inner")

        yearly_dfs.append(df_year)

    df_era5 = pd.concat(yearly_dfs).sort_index()
    df_era5 = df_era5.loc[~df_era5.index.duplicated(keep="first")]

    if df_era5.index.tz is None:
        df_era5.index = df_era5.index.tz_localize("UTC")

    df_era5.index = df_era5.index.tz_convert(target_tz)
    df_era5.index.name = "timestamp"
    return df_era5


def load_dwd(
    icon_dir: Path,
    start_folder_date: date,
    required_run: str,
    skip_dates: set[date] = frozenset(),
    folder_offset_date: date = date(2025, 10, 26),
    target_tz: str = "Europe/Berlin",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load clustered DWD ICON-D2 forecast data from the daily folder structure."""
    if not icon_dir.exists():
        raise FileNotFoundError(f"ICON directory not found: {icon_dir}")

    hourly_dfs = []
    qh_dfs = []

    for folder_name in sorted(os.listdir(icon_dir)):
        folder_path = icon_dir / folder_name
        if not folder_path.is_dir():
            continue

        try:
            folder_parts = folder_name.split("_")
            date_str = folder_parts[3]
            folder_date = datetime.strptime(date_str, "%Y%m%d").date()
        except Exception:
            continue

        folder_run = ""
        if len(folder_parts) > 4 and folder_parts[4].isdigit() and len(folder_parts[4]) == 2:
            folder_run = folder_parts[4]
        if folder_run and folder_run != required_run:
            continue

        if folder_date < start_folder_date or folder_date in skip_dates:
            continue

        forecast_date = folder_date if folder_date < folder_offset_date else folder_date + timedelta(days=1)
        variable_dfs = []

        for filename in sorted(os.listdir(folder_path)):
            if not filename.endswith(".csv"):
                continue

            parts = filename.replace(".csv", "").split("_")
            run_hour = next((token[-2:] for token in parts if token.isdigit() and len(token) == 10), "")
            if not folder_run and run_hour != required_run:
                continue

            file_path = folder_path / filename
            var_name = _canonical_dwd_variable_name(_read_dwd_variable_name(file_path))
            df_var = pd.read_csv(file_path, comment="#", sep=",", engine="python")
            if "timestamp" not in df_var.columns:
                raise ValueError(f"Missing 'timestamp' column in: {filename}")

            df_var = df_var.rename(columns={col: f"{var_name}_{col}" for col in df_var.columns if col.startswith("cluster_")})
            df_var["timestamp"] = pd.to_datetime(df_var["timestamp"], utc=True).dt.tz_convert(target_tz)
            variable_dfs.append(df_var)

        if not variable_dfs:
            raise ValueError(f"No CSVs with run={required_run} found in: {folder_name}")

        df_day = variable_dfs[0].copy()
        for df_next in variable_dfs[1:]:
            df_day = df_day.merge(df_next, on="timestamp", how="outer")
        df_day = df_day.sort_values("timestamp")

        start = pd.Timestamp(forecast_date, tz=target_tz)
        end = start + pd.Timedelta(days=1)

        hourly_cols = ["timestamp"] + [
            col
            for col in df_day.columns
            if col.startswith(
                (
                    "t2m_",
                    "td2m_",
                    "u10_",
                    "v10_",
                    "u_ml",
                    "v_ml",
                    "vmax10m_",
                    "sp_",
                    "tp_",
                    "sde_",
                    "snow_gsp_",
                    "tcc_",
                    "lcc_",
                    "mcc_",
                    "hcc_",
                )
            )
        ]
        qh_cols = ["timestamp"] + [col for col in df_day.columns if col.startswith(("ASWDIR_", "ASWDIFD_"))]

        df_hourly_day = (
            df_day[hourly_cols]
            .loc[
                (df_day["timestamp"] >= start)
                & (df_day["timestamp"] < end)
                & (df_day["timestamp"].dt.minute == 0)
            ]
            .copy()
        )

        df_qh_day = df_day[qh_cols].copy()
        df_qh_day["timestamp"] = df_qh_day["timestamp"] - pd.Timedelta(minutes=15)
        df_qh_day = df_qh_day.loc[
            (df_qh_day["timestamp"] >= start)
            & (df_qh_day["timestamp"] < end)
        ]

        hourly_dfs.append(df_hourly_day)
        qh_dfs.append(df_qh_day)

    if not hourly_dfs:
        raise ValueError("No hourly DWD data loaded.")
    if not qh_dfs:
        raise ValueError("No quarter-hourly DWD data loaded.")

    df_hourly = pd.concat(hourly_dfs, ignore_index=True).sort_values("timestamp").set_index("timestamp")
    df_qh = pd.concat(qh_dfs, ignore_index=True).sort_values("timestamp").set_index("timestamp")

    df_hourly.index.name = "timestamp"
    df_qh.index.name = "timestamp"
    return df_hourly, df_qh


def _direction_to_uv(speed_m_s: pd.Series, direction_degrees: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Convert meteorological wind direction to eastward/northward components."""
    radians = np.deg2rad(direction_degrees.astype(float))
    speed = speed_m_s.astype(float)
    u = -speed * np.sin(radians)
    v = -speed * np.cos(radians)
    return u, v


def _interpolate_100m_wind(df: pd.DataFrame) -> pd.DataFrame:
    """Build wind vector components from Open-Meteo speed/direction pairs."""
    if {"wind_speed_10m", "wind_direction_10m"} <= set(df.columns):
        df["u10"], df["v10"] = _direction_to_uv(df["wind_speed_10m"], df["wind_direction_10m"])
    has_80 = {"wind_speed_80m", "wind_direction_80m"} <= set(df.columns)
    has_120 = {"wind_speed_120m", "wind_direction_120m"} <= set(df.columns)

    if has_80:
        df["u80"], df["v80"] = _direction_to_uv(df["wind_speed_80m"], df["wind_direction_80m"])
    if {"wind_speed_100m", "wind_direction_100m"} <= set(df.columns):
        df["u100"], df["v100"] = _direction_to_uv(df["wind_speed_100m"], df["wind_direction_100m"])
    if has_120:
        df["u120"], df["v120"] = _direction_to_uv(df["wind_speed_120m"], df["wind_direction_120m"])
    if {"wind_speed_180m", "wind_direction_180m"} <= set(df.columns):
        df["u180"], df["v180"] = _direction_to_uv(df["wind_speed_180m"], df["wind_direction_180m"])

    if "u100" not in df.columns and has_80 and has_120:
        weight = (100.0 - 80.0) / (120.0 - 80.0)
        df["u100"] = df["u80"] + weight * (df["u120"] - df["u80"])
        df["v100"] = df["v80"] + weight * (df["v120"] - df["v80"])
    return df


def _normalise_open_meteo_units(df: pd.DataFrame) -> pd.DataFrame:
    """Convert Open-Meteo units to the conventions used by the regional feature builder."""
    df = df.copy()
    if "temperature_2m" in df.columns:
        df["t2m"] = df["temperature_2m"] + 273.15
    if "dew_point_2m" in df.columns:
        df["td2m"] = df["dew_point_2m"] + 273.15
    if "temperature_80m" in df.columns:
        df["t80m"] = df["temperature_80m"] + 273.15
    if "temperature_100m" in df.columns:
        df["t100m"] = df["temperature_100m"] + 273.15
    if "temperature_120m" in df.columns:
        df["t120m"] = df["temperature_120m"] + 273.15
    if "temperature_180m" in df.columns:
        df["t180m"] = df["temperature_180m"] + 273.15
    if "relative_humidity_2m" in df.columns:
        df["r2m"] = df["relative_humidity_2m"]
    if "vapour_pressure_deficit" in df.columns:
        df["vpd"] = df["vapour_pressure_deficit"]
    if "surface_pressure" in df.columns:
        # Open-Meteo returns surface pressure in hPa.
        df["sp"] = df["surface_pressure"] * 100.0
    if "pressure_msl" in df.columns:
        # Open-Meteo returns sea-level pressure in hPa.
        df["msl"] = df["pressure_msl"] * 100.0
    if "shortwave_radiation" in df.columns:
        df["ssrd"] = df["shortwave_radiation"]
    elif {"direct_radiation", "diffuse_radiation"} <= set(df.columns):
        df["ssrd"] = df["direct_radiation"] + df["diffuse_radiation"]
    if "direct_radiation" in df.columns:
        df["fdir"] = df["direct_radiation"]
    if "direct_normal_irradiance" in df.columns:
        df["dni"] = df["direct_normal_irradiance"]
    if "diffuse_radiation" in df.columns:
        df["diffuse"] = df["diffuse_radiation"]
    if "cloud_cover" in df.columns:
        df["tcc"] = df["cloud_cover"]
    if "cloud_cover_low" in df.columns:
        df["lcc"] = df["cloud_cover_low"]
    if "cloud_cover_mid" in df.columns:
        df["mcc"] = df["cloud_cover_mid"]
    if "cloud_cover_high" in df.columns:
        df["hcc"] = df["cloud_cover_high"]
    if "precipitation" in df.columns:
        df["tp"] = df["precipitation"]
    if "wind_gusts_10m" in df.columns:
        df["vmax10m"] = df["wind_gusts_10m"]
    if "snow_depth" in df.columns:
        df["sde"] = df["snow_depth"]
    elif "snow_height" in df.columns:
        df["sde"] = df["snow_height"]
    if "boundary_layer_height" in df.columns:
        df["pblh"] = df["boundary_layer_height"]

    df = _interpolate_100m_wind(df)
    return df


_OPEN_METEO_NORMALISED_COLUMNS = [
    "u80",
    "v80",
    "u100",
    "v100",
    "u120",
    "v120",
    "u180",
    "v180",
    "u10",
    "v10",
    "t2m",
    "td2m",
    "t80m",
    "t100m",
    "t120m",
    "t180m",
    "r2m",
    "vpd",
    "sp",
    "msl",
    "ssrd",
    "fdir",
    "dni",
    "diffuse",
    "tcc",
    "lcc",
    "mcc",
    "hcc",
    "tp",
    "vmax10m",
    "sde",
    "pblh",
    "cape",
    "sunshine_duration",
]


def _normalise_open_meteo_points(points: pd.DataFrame) -> pd.DataFrame:
    required = {"weather_point_id", "lat", "lon"}
    missing = required - set(points.columns)
    if missing:
        raise ValueError(f"Open-Meteo point table is missing required columns: {sorted(missing)}")

    result = points[["weather_point_id", "lat", "lon"]].dropna().copy()
    result["weather_point_id"] = result["weather_point_id"].astype(int)
    result["lat"] = result["lat"].astype(float)
    result["lon"] = result["lon"].astype(float)
    result = result.sort_values("weather_point_id").reset_index(drop=True)
    if result["weather_point_id"].duplicated().any():
        raise ValueError("weather_point_id values must be unique.")
    if result.empty:
        raise ValueError("Open-Meteo point table contains no usable points.")
    return result


def _open_meteo_point_keep_columns(point_id: int) -> dict[str, str]:
    return {column: f"{column}_point_{point_id}" for column in _OPEN_METEO_NORMALISED_COLUMNS}


def _open_meteo_response_items(payload: object) -> list[dict]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        return [payload]
    raise ValueError(f"Unexpected Open-Meteo response type: {type(payload)!r}")


def _fetch_open_meteo_batch(
    *,
    base_url: str,
    latitude: list[float],
    longitude: list[float],
    hourly_variables: list[str],
    model: str | None,
    cell_selection: str,
    timeout_seconds: int,
    start_date: date | None = None,
    end_date: date | None = None,
    run: str | None = None,
    forecast_days: int | None = None,
    retry_attempts: int = 5,
    retry_backoff_seconds: float = 30.0,
    api_key_env: str | None = None,
) -> list[dict]:
    params: dict[str, str] = {
        "latitude": ",".join(f"{value:.6f}" for value in latitude),
        "longitude": ",".join(f"{value:.6f}" for value in longitude),
        "hourly": ",".join(hourly_variables),
        "timezone": "UTC",
        "wind_speed_unit": "ms",
        "cell_selection": cell_selection,
    }
    if run is not None:
        params["run"] = run
        if forecast_days is not None:
            params["forecast_days"] = str(forecast_days)
    else:
        if start_date is None or end_date is None:
            raise ValueError("start_date and end_date are required unless a single model run is requested.")
        params["start_date"] = start_date.isoformat()
        params["end_date"] = end_date.isoformat()

    if model:
        params["models"] = model
    if api_key_env:
        api_key = os.getenv(api_key_env)
        if api_key:
            params["apikey"] = api_key
        elif "customer-" in base_url:
            fallback_url = base_url.replace("://customer-", "://", 1)
            print(
                f"[OPEN-METEO] Customer endpoint configured but environment variable "
                f"{api_key_env!r} is not set. Falling back to {fallback_url}."
            )
            base_url = fallback_url

    last_error: Exception | None = None
    response: requests.Response | None = None
    for attempt in range(retry_attempts + 1):
        try:
            response = requests.get(base_url, params=params, timeout=timeout_seconds)
        except requests.RequestException as exc:
            last_error = exc
            if attempt >= retry_attempts:
                raise

            sleep_seconds = retry_backoff_seconds * (attempt + 1)
            print(
                f"[OPEN-METEO] Request failed with {type(exc).__name__}. "
                f"Retrying in {sleep_seconds:.0f}s ({attempt + 1}/{retry_attempts})..."
            )
            time.sleep(sleep_seconds)
            continue

        if response.status_code != 429:
            if response.status_code >= 400:
                _raise_open_meteo_http_error(response)
            try:
                payload = response.json()
            except ValueError as exc:
                if _is_open_meteo_model_run_unavailable(response.text):
                    raise OpenMeteoModelRunUnavailable(response.text[:500])
                last_error = requests.RequestException(
                    f"Open-Meteo returned invalid JSON (body={response.text[:80]!r}): {exc}"
                )
                if attempt >= retry_attempts:
                    raise last_error
                sleep_seconds = retry_backoff_seconds * (attempt + 1)
                print(
                    f"[OPEN-METEO] Invalid JSON response. "
                    f"Retrying in {sleep_seconds:.0f}s ({attempt + 1}/{retry_attempts})..."
                )
                time.sleep(sleep_seconds)
                continue
            if isinstance(payload, dict) and "error" in payload:
                reason = payload.get("reason", payload)
                if _is_open_meteo_model_run_unavailable(reason):
                    raise OpenMeteoModelRunUnavailable(str(reason)[:500])
                raise RuntimeError(f"Open-Meteo error: {reason}")
            return _open_meteo_response_items(payload)

        last_error = requests.HTTPError(
            f"429 Client Error: Too Many Requests for url: {response.url}",
            response=response,
        )
        if attempt >= retry_attempts:
            raise last_error

        retry_after = response.headers.get("Retry-After")
        if retry_after is not None:
            try:
                sleep_seconds = float(retry_after)
            except ValueError:
                sleep_seconds = retry_backoff_seconds * (attempt + 1)
        else:
            sleep_seconds = retry_backoff_seconds * (attempt + 1)

        print(
            f"[OPEN-METEO] Rate limited (429). "
            f"Retrying in {sleep_seconds:.0f}s ({attempt + 1}/{retry_attempts})..."
        )
        time.sleep(sleep_seconds)
    else:
        if last_error is not None:
            raise last_error

    if response is None:
        raise RuntimeError("Open-Meteo request failed without a response.")
    raise RuntimeError("Open-Meteo request loop exited without returning a payload.")


def fetch_open_meteo_cluster_weather(
    *,
    cluster_file: Path,
    start_date: date,
    end_date: date,
    output_file: Path | None = None,
    base_url: str = "https://customer-historical-forecast-api.open-meteo.com/v1/forecast",
    model: str | None = "icon_d2",
    hourly_variables: list[str] | None = None,
    batch_size: int = 10,
    cell_selection: str = "nearest",
    timeout_seconds: int = 60,
    target_tz: str = "Europe/Berlin",
    point_selection: str = "centroid",
    max_points_per_cluster: int | None = None,
    api_mode: str = "historical_forecast",
    single_run_hour_utc: str = "06:00",
    single_run_forecast_days: int = 2,
    request_pause_seconds: float = 0.0,
    retry_attempts: int = 5,
    retry_backoff_seconds: float = 30.0,
    api_key_env: str | None = None,
    skip_unavailable_runs: bool = False,
    fallback_previous_runs: bool = False,
    fallback_step_hours: int = 2,
    fallback_max_lookback_hours: int = 24,
) -> pd.DataFrame:
    """Fetch Open-Meteo weather at cluster centroids and return ICON-style columns."""
    if hourly_variables is None:
        hourly_variables = [
            "wind_speed_80m",
            "wind_direction_80m",
            "wind_speed_120m",
            "wind_direction_120m",
            "temperature_2m",
            "dew_point_2m",
            "surface_pressure",
            "shortwave_radiation",
            "direct_radiation",
            "diffuse_radiation",
            "cloud_cover",
            "precipitation",
            "snow_depth",
        ]

    if cluster_file.suffix.lower() == ".parquet":
        clusters = pd.read_parquet(cluster_file)
    else:
        clusters = pd.read_csv(cluster_file)

    required = {"cluster_id", "lat", "lon"}
    missing = required - set(clusters.columns)
    if missing:
        raise ValueError(f"Cluster file is missing required columns: {sorted(missing)}")

    clusters = clusters[["cluster_id", "lat", "lon"]].dropna().copy()
    clusters["cluster_id"] = clusters["cluster_id"].astype(int)
    clusters = clusters.sort_values(["cluster_id", "lat", "lon"]).reset_index(drop=True)

    if clusters.empty:
        raise ValueError(f"No cluster coordinates found in: {cluster_file}")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    if point_selection == "centroid":
        points = (
            clusters
            .groupby("cluster_id", as_index=False)[["lat", "lon"]]
            .mean()
            .sort_values("cluster_id")
            .reset_index(drop=True)
        )
        points["point_id"] = 0
    elif point_selection == "grid_mean":
        points = clusters.copy()
        if max_points_per_cluster is not None:
            if max_points_per_cluster <= 0:
                raise ValueError("max_points_per_cluster must be positive when provided.")
            sampled_groups = []
            for _, group in points.groupby("cluster_id", sort=True):
                selected = np.linspace(
                    0,
                    len(group) - 1,
                    min(max_points_per_cluster, len(group)),
                    dtype=int,
                )
                sampled_groups.append(group.iloc[selected].copy())
            points = pd.concat(sampled_groups, ignore_index=True)
        points["point_id"] = points.groupby("cluster_id").cumcount()
    else:
        raise ValueError("point_selection must be either 'centroid' or 'grid_mean'.")

    frames = []
    if api_mode == "historical_forecast":
        request_start_date = start_date - timedelta(days=1)
        request_end_date = end_date + timedelta(days=1)
        requests_to_make: list[tuple[pd.Timestamp | None, str | None]] = [(None, None)]
    elif api_mode == "single_run":
        request_start_date = None
        request_end_date = None
        target_days = pd.date_range(
            start=pd.Timestamp(start_date, tz=target_tz),
            end=pd.Timestamp(end_date, tz=target_tz),
            freq="D",
        )
        requests_to_make = []
        for target_day in target_days:
            run_date = target_day.date() - timedelta(days=1)
            requests_to_make.append((target_day, f"{run_date.isoformat()}T{single_run_hour_utc}"))
    else:
        raise ValueError("api_mode must be either 'historical_forecast' or 'single_run'.")

    completed_weather: pd.DataFrame | None = None
    completed_days: set[pd.Timestamp] = set()
    if api_mode == "single_run" and output_file is not None and output_file.exists():
        completed_weather = pd.read_csv(output_file, index_col=0)
        completed_weather.index = pd.to_datetime(completed_weather.index, utc=True).tz_convert(target_tz)
        completed_weather = completed_weather.sort_index()
        completed_weather = completed_weather.loc[~completed_weather.index.duplicated(keep="last")]
        completed_weather = completed_weather.loc[:, ~completed_weather.columns.duplicated()]
        completed_days = {pd.Timestamp(value).normalize() for value in completed_weather.index.normalize().unique()}

    for target_day, run in requests_to_make:
        if target_day is not None and target_day.normalize() in completed_days:
            print(f"[OPEN-METEO] Skipping cached day: {target_day.date()}")
            continue

        day_frames_start = len(frames)
        day_unavailable = False
        day_run = run
        day_forecast_days = single_run_forecast_days if run is not None else None
        for start in range(0, len(points), batch_size):
            batch = points.iloc[start:start + batch_size]
            try:
                items, day_run, day_forecast_days = _fetch_open_meteo_batch_with_run_fallback(
                    base_url=base_url,
                    latitude=batch["lat"].astype(float).tolist(),
                    longitude=batch["lon"].astype(float).tolist(),
                    start_date=request_start_date,
                    end_date=request_end_date,
                    target_day=target_day,
                    run=day_run,
                    forecast_days=day_forecast_days,
                    hourly_variables=hourly_variables,
                    model=model,
                    cell_selection=cell_selection,
                    timeout_seconds=timeout_seconds,
                    retry_attempts=retry_attempts,
                    retry_backoff_seconds=retry_backoff_seconds,
                    api_key_env=api_key_env,
                    fallback_previous_runs=fallback_previous_runs,
                    fallback_step_hours=fallback_step_hours,
                    fallback_max_lookback_hours=fallback_max_lookback_hours,
                )
            except OpenMeteoModelRunUnavailable as exc:
                if not skip_unavailable_runs:
                    raise
                day_label = target_day.date() if target_day is not None else request_start_date
                print(f"[OPEN-METEO] Skipping unavailable model run for day {day_label}: {exc}")
                del frames[day_frames_start:]
                day_unavailable = True
                break
            if request_pause_seconds > 0:
                time.sleep(request_pause_seconds)

            if len(items) != len(batch):
                raise ValueError(
                    "Open-Meteo returned a different number of locations than requested "
                    f"({len(items)} returned, {len(batch)} requested)."
                )

            for (_, point_row), item in zip(batch.iterrows(), items, strict=True):
                hourly = item.get("hourly")
                if not isinstance(hourly, dict) or "time" not in hourly:
                    raise ValueError(
                        f"Open-Meteo response is missing hourly time series for cluster {point_row.cluster_id}."
                    )

                df_cluster = pd.DataFrame(hourly)
                df_cluster["timestamp"] = pd.to_datetime(df_cluster["time"], utc=True).dt.tz_convert(target_tz)
                df_cluster = df_cluster.drop(columns=["time"]).set_index("timestamp")
                if target_day is not None:
                    target_start = target_day
                    target_end = target_start + pd.Timedelta(days=1)
                    df_cluster = df_cluster.loc[(df_cluster.index >= target_start) & (df_cluster.index < target_end)]
                df_cluster = _normalise_open_meteo_units(df_cluster)

                cluster_id = int(point_row.cluster_id)
                point_id = int(point_row.point_id)
                keep_columns = {
                    column: f"{column}_cluster_{cluster_id}_point_{point_id}"
                    for column in _OPEN_METEO_NORMALISED_COLUMNS
                }
                available = [column for column in keep_columns if column in df_cluster.columns]
                frames.append(df_cluster[available].rename(columns=keep_columns))

        if day_unavailable:
            continue

        if api_mode == "single_run" and target_day is not None and output_file is not None:
            day_weather = _average_open_meteo_point_frames(frames[day_frames_start:], points)
            if completed_weather is not None:
                completed_weather = pd.concat([completed_weather, day_weather]).sort_index()
                completed_weather = completed_weather.loc[~completed_weather.index.duplicated(keep="last")]
            else:
                completed_weather = day_weather
            output_file.parent.mkdir(parents=True, exist_ok=True)
            completed_weather.to_csv(output_file)
            completed_days.add(target_day.normalize())
            print(f"[OPEN-METEO] Cached single-run day: {target_day.date()}")

    if api_mode == "single_run" and completed_weather is not None and output_file is not None:
        start_ts = pd.Timestamp(start_date, tz=target_tz)
        end_ts = pd.Timestamp(end_date + timedelta(days=1), tz=target_tz)
        return completed_weather.loc[(completed_weather.index >= start_ts) & (completed_weather.index < end_ts)]

    if not frames:
        if completed_weather is not None:
            start_ts = pd.Timestamp(start_date, tz=target_tz)
            end_ts = pd.Timestamp(end_date + timedelta(days=1), tz=target_tz)
            return completed_weather.loc[(completed_weather.index >= start_ts) & (completed_weather.index < end_ts)]
        raise ValueError("No Open-Meteo weather data was fetched.")

    weather = _average_open_meteo_point_frames(frames, points)
    if completed_weather is not None:
        weather = pd.concat([completed_weather, weather]).sort_index()
        weather = weather.loc[~weather.index.duplicated(keep="last")]

    start_ts = pd.Timestamp(start_date, tz=target_tz)
    end_ts = pd.Timestamp(end_date + timedelta(days=1), tz=target_tz)
    weather = weather.loc[(weather.index >= start_ts) & (weather.index < end_ts)]

    if output_file is not None:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        if api_mode == "single_run" and completed_weather is not None:
            completed_weather.to_csv(output_file)
        else:
            weather.to_csv(output_file)

    return weather


def _average_open_meteo_point_frames(frames: list[pd.DataFrame], points: pd.DataFrame) -> pd.DataFrame:
    """Average point-level Open-Meteo columns back to cluster-level weather columns."""
    weather = pd.concat(frames, axis=1).sort_index()
    weather = weather.loc[~weather.index.duplicated(keep="last")]

    averaged = {}
    for base_name in _OPEN_METEO_NORMALISED_COLUMNS:
        for cluster_id in sorted(points["cluster_id"].unique()):
            prefix = f"{base_name}_cluster_{cluster_id}_point_"
            columns = [column for column in weather.columns if column.startswith(prefix)]
            if columns:
                averaged[f"{base_name}_cluster_{cluster_id}"] = weather[columns].mean(axis=1)

    weather = pd.DataFrame(averaged, index=weather.index)
    weather.index.name = "timestamp"
    return weather


def fetch_open_meteo_point_weather(
    *,
    points: pd.DataFrame,
    start_date: date,
    end_date: date,
    output_file: Path | None = None,
    base_url: str = "https://customer-historical-forecast-api.open-meteo.com/v1/forecast",
    model: str | None = "icon_d2",
    hourly_variables: list[str] | None = None,
    batch_size: int = 10,
    cell_selection: str = "nearest",
    timeout_seconds: int = 60,
    target_tz: str = "Europe/Berlin",
    api_mode: str = "historical_forecast",
    single_run_hour_utc: str = "06:00",
    single_run_forecast_days: int = 2,
    request_pause_seconds: float = 0.0,
    retry_attempts: int = 5,
    retry_backoff_seconds: float = 30.0,
    api_key_env: str | None = None,
    skip_unavailable_runs: bool = False,
    fallback_previous_runs: bool = False,
    fallback_step_hours: int = 2,
    fallback_max_lookback_hours: int = 24,
) -> pd.DataFrame:
    """Fetch Open-Meteo weather for fixed representative points and keep point-level columns."""
    if hourly_variables is None:
        hourly_variables = [
            "wind_speed_80m",
            "wind_direction_80m",
            "wind_speed_120m",
            "wind_direction_120m",
            "temperature_2m",
            "dew_point_2m",
            "surface_pressure",
            "shortwave_radiation",
            "direct_radiation",
            "diffuse_radiation",
            "cloud_cover",
            "precipitation",
            "snow_depth",
        ]

    points = _normalise_open_meteo_points(points)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    frames = []
    if api_mode == "historical_forecast":
        request_start_date = start_date - timedelta(days=1)
        request_end_date = end_date + timedelta(days=1)
        requests_to_make: list[tuple[pd.Timestamp | None, str | None]] = [(None, None)]
    elif api_mode == "single_run":
        request_start_date = None
        request_end_date = None
        target_days = pd.date_range(
            start=pd.Timestamp(start_date, tz=target_tz),
            end=pd.Timestamp(end_date, tz=target_tz),
            freq="D",
        )
        requests_to_make = []
        for target_day in target_days:
            run_date = target_day.date() - timedelta(days=1)
            requests_to_make.append((target_day, f"{run_date.isoformat()}T{single_run_hour_utc}"))
    else:
        raise ValueError("api_mode must be either 'historical_forecast' or 'single_run'.")

    completed_weather: pd.DataFrame | None = None
    completed_days: set[pd.Timestamp] = set()
    if api_mode == "single_run" and output_file is not None and output_file.exists():
        completed_weather = pd.read_csv(output_file, index_col=0)
        completed_weather.index = pd.to_datetime(completed_weather.index, utc=True).tz_convert(target_tz)
        completed_weather = completed_weather.sort_index()
        completed_weather = completed_weather.loc[~completed_weather.index.duplicated(keep="last")]
        completed_weather = completed_weather.loc[:, ~completed_weather.columns.duplicated()]
        completed_days = {pd.Timestamp(value).normalize() for value in completed_weather.index.normalize().unique()}

    for target_day, run in requests_to_make:
        if target_day is not None and target_day.normalize() in completed_days:
            print(f"[OPEN-METEO] Skipping cached point-weather day: {target_day.date()}")
            continue

        day_frames_start = len(frames)
        day_unavailable = False
        day_run = run
        day_forecast_days = single_run_forecast_days if run is not None else None
        for start in range(0, len(points), batch_size):
            batch = points.iloc[start:start + batch_size]
            try:
                items, day_run, day_forecast_days = _fetch_open_meteo_batch_with_run_fallback(
                    base_url=base_url,
                    latitude=batch["lat"].astype(float).tolist(),
                    longitude=batch["lon"].astype(float).tolist(),
                    start_date=request_start_date,
                    end_date=request_end_date,
                    target_day=target_day,
                    run=day_run,
                    forecast_days=day_forecast_days,
                    hourly_variables=hourly_variables,
                    model=model,
                    cell_selection=cell_selection,
                    timeout_seconds=timeout_seconds,
                    retry_attempts=retry_attempts,
                    retry_backoff_seconds=retry_backoff_seconds,
                    api_key_env=api_key_env,
                    fallback_previous_runs=fallback_previous_runs,
                    fallback_step_hours=fallback_step_hours,
                    fallback_max_lookback_hours=fallback_max_lookback_hours,
                )
            except OpenMeteoModelRunUnavailable as exc:
                if not skip_unavailable_runs:
                    raise
                day_label = target_day.date() if target_day is not None else request_start_date
                print(f"[OPEN-METEO] Skipping unavailable model run for day {day_label}: {exc}")
                del frames[day_frames_start:]
                day_unavailable = True
                break
            if request_pause_seconds > 0:
                time.sleep(request_pause_seconds)

            if len(items) != len(batch):
                raise ValueError(
                    "Open-Meteo returned a different number of locations than requested "
                    f"({len(items)} returned, {len(batch)} requested)."
                )

            for (_, point_row), item in zip(batch.iterrows(), items, strict=True):
                hourly = item.get("hourly")
                if not isinstance(hourly, dict) or "time" not in hourly:
                    raise ValueError(
                        f"Open-Meteo response is missing hourly time series for point {point_row.weather_point_id}."
                    )

                df_point = pd.DataFrame(hourly)
                df_point["timestamp"] = pd.to_datetime(df_point["time"], utc=True).dt.tz_convert(target_tz)
                df_point = df_point.drop(columns=["time"]).set_index("timestamp")
                if target_day is not None:
                    target_start = target_day
                    target_end = target_start + pd.Timedelta(days=1)
                    df_point = df_point.loc[(df_point.index >= target_start) & (df_point.index < target_end)]
                df_point = _normalise_open_meteo_units(df_point)

                point_id = int(point_row.weather_point_id)
                keep_columns = _open_meteo_point_keep_columns(point_id)
                available = [column for column in keep_columns if column in df_point.columns]
                frames.append(df_point[available].rename(columns=keep_columns))

        if day_unavailable:
            continue

        if api_mode == "single_run" and target_day is not None and output_file is not None:
            day_weather = pd.concat(frames[day_frames_start:], axis=1).sort_index()
            day_weather = day_weather.loc[~day_weather.index.duplicated(keep="last")]
            day_weather.index.name = "timestamp"
            if completed_weather is not None:
                completed_weather = pd.concat([completed_weather, day_weather]).sort_index()
                completed_weather = completed_weather.loc[~completed_weather.index.duplicated(keep="last")]
            else:
                completed_weather = day_weather
            output_file.parent.mkdir(parents=True, exist_ok=True)
            completed_weather.to_csv(output_file)
            completed_days.add(target_day.normalize())
            print(f"[OPEN-METEO] Cached single-run point-weather day: {target_day.date()}")

    if api_mode == "single_run" and completed_weather is not None and output_file is not None:
        return completed_weather

    if not frames:
        if completed_weather is not None:
            if api_mode == "single_run":
                return completed_weather
            start_ts = pd.Timestamp(start_date, tz=target_tz)
            end_ts = pd.Timestamp(end_date + timedelta(days=1), tz=target_tz)
            return completed_weather.loc[(completed_weather.index >= start_ts) & (completed_weather.index < end_ts)]
        raise ValueError("No Open-Meteo point weather data was fetched.")

    weather = pd.concat(frames, axis=1).sort_index()
    weather = weather.loc[~weather.index.duplicated(keep="last")]
    weather.index.name = "timestamp"
    if completed_weather is not None:
        weather = pd.concat([completed_weather, weather]).sort_index()
        weather = weather.loc[~weather.index.duplicated(keep="last")]

    if api_mode != "single_run":
        start_ts = pd.Timestamp(start_date, tz=target_tz)
        end_ts = pd.Timestamp(end_date + timedelta(days=1), tz=target_tz)
        weather = weather.loc[(weather.index >= start_ts) & (weather.index < end_ts)]

    if output_file is not None:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        weather.to_csv(output_file)

    return weather


def load_open_meteo_points(
    *,
    points: pd.DataFrame,
    start_date: date,
    end_date: date,
    cache_file: Path,
    base_url: str = "https://customer-historical-forecast-api.open-meteo.com/v1/forecast",
    model: str | None = "icon_d2",
    hourly_variables: list[str] | None = None,
    batch_size: int = 10,
    cell_selection: str = "nearest",
    timeout_seconds: int = 60,
    target_tz: str = "Europe/Berlin",
    force_download: bool = False,
    api_mode: str = "historical_forecast",
    single_run_hour_utc: str = "06:00",
    single_run_forecast_days: int = 2,
    request_pause_seconds: float = 0.0,
    retry_attempts: int = 5,
    retry_backoff_seconds: float = 30.0,
    api_key_env: str | None = None,
    skip_unavailable_runs: bool = False,
    fallback_previous_runs: bool = False,
    fallback_step_hours: int = 2,
    fallback_max_lookback_hours: int = 24,
) -> pd.DataFrame:
    """Load cached point-level Open-Meteo weather or fetch missing single-run days."""
    if cache_file.exists() and not force_download:
        df = pd.read_csv(cache_file, index_col=0)
        df.index = pd.to_datetime(df.index, utc=True).tz_convert(target_tz)
        df = df.sort_index()
        df = df.loc[~df.index.duplicated(keep="last")]
        df.index.name = "timestamp"
        if api_mode != "single_run":
            return df

        expected_days = pd.date_range(
            start=pd.Timestamp(start_date, tz=target_tz),
            end=pd.Timestamp(end_date, tz=target_tz),
            freq="D",
        )
        cached_days = pd.DatetimeIndex(df.index.normalize().unique()).sort_values()
        missing_days = expected_days.difference(cached_days)
        if missing_days.empty:
            return df

        print(
            "[OPEN-METEO] Point-weather single-run cache is incomplete: "
            f"{len(cached_days)}/{len(expected_days)} days cached. "
            f"Resuming from {missing_days.min().date()}..."
        )

    return fetch_open_meteo_point_weather(
        points=points,
        start_date=start_date,
        end_date=end_date,
        output_file=cache_file,
        base_url=base_url,
        model=model,
        hourly_variables=hourly_variables,
        batch_size=batch_size,
        cell_selection=cell_selection,
        timeout_seconds=timeout_seconds,
        target_tz=target_tz,
        api_mode=api_mode,
        single_run_hour_utc=single_run_hour_utc,
        single_run_forecast_days=single_run_forecast_days,
        request_pause_seconds=request_pause_seconds,
        retry_attempts=retry_attempts,
        retry_backoff_seconds=retry_backoff_seconds,
        api_key_env=api_key_env,
        skip_unavailable_runs=skip_unavailable_runs,
        fallback_previous_runs=fallback_previous_runs,
        fallback_step_hours=fallback_step_hours,
        fallback_max_lookback_hours=fallback_max_lookback_hours,
    )


def load_open_meteo(
    *,
    cluster_file: Path,
    start_date: date,
    end_date: date,
    cache_file: Path,
    base_url: str = "https://customer-historical-forecast-api.open-meteo.com/v1/forecast",
    model: str | None = "icon_d2",
    hourly_variables: list[str] | None = None,
    batch_size: int = 10,
    cell_selection: str = "nearest",
    timeout_seconds: int = 60,
    target_tz: str = "Europe/Berlin",
    force_download: bool = False,
    point_selection: str = "centroid",
    max_points_per_cluster: int | None = None,
    api_mode: str = "historical_forecast",
    single_run_hour_utc: str = "06:00",
    single_run_forecast_days: int = 2,
    request_pause_seconds: float = 0.0,
    retry_attempts: int = 5,
    retry_backoff_seconds: float = 30.0,
    api_key_env: str | None = None,
    skip_unavailable_runs: bool = False,
    fallback_previous_runs: bool = False,
    fallback_step_hours: int = 2,
    fallback_max_lookback_hours: int = 24,
) -> pd.DataFrame:
    """Load cached Open-Meteo cluster weather or fetch it from the API."""
    if cache_file.exists() and not force_download:
        df = pd.read_csv(cache_file, index_col=0)
        df.index = pd.to_datetime(df.index, utc=True).tz_convert(target_tz)
        df = df.sort_index()
        df = df.loc[~df.index.duplicated(keep="last")]
        df.index.name = "timestamp"
        if api_mode != "single_run":
            return df

        expected_days = pd.date_range(
            start=pd.Timestamp(start_date, tz=target_tz),
            end=pd.Timestamp(end_date, tz=target_tz),
            freq="D",
        )
        cached_days = pd.DatetimeIndex(df.index.normalize().unique()).sort_values()
        missing_days = expected_days.difference(cached_days)
        if missing_days.empty:
            return df

        print(
            "[OPEN-METEO] Single-run cache is incomplete: "
            f"{len(cached_days)}/{len(expected_days)} days cached. "
            f"Resuming from {missing_days.min().date()}..."
        )

    return fetch_open_meteo_cluster_weather(
        cluster_file=cluster_file,
        start_date=start_date,
        end_date=end_date,
        output_file=cache_file,
        base_url=base_url,
        model=model,
        hourly_variables=hourly_variables,
        batch_size=batch_size,
        cell_selection=cell_selection,
        timeout_seconds=timeout_seconds,
        target_tz=target_tz,
        point_selection=point_selection,
        max_points_per_cluster=max_points_per_cluster,
        api_mode=api_mode,
        single_run_hour_utc=single_run_hour_utc,
        single_run_forecast_days=single_run_forecast_days,
        request_pause_seconds=request_pause_seconds,
        retry_attempts=retry_attempts,
        retry_backoff_seconds=retry_backoff_seconds,
        api_key_env=api_key_env,
        skip_unavailable_runs=skip_unavailable_runs,
        fallback_previous_runs=fallback_previous_runs,
        fallback_step_hours=fallback_step_hours,
        fallback_max_lookback_hours=fallback_max_lookback_hours,
    )
