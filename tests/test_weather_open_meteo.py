from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest
import requests

from da_price_forecasting.data import weather
from da_price_forecasting.data.weather import (
    OpenMeteoModelRunUnavailable,
    _fetch_open_meteo_batch,
    _normalise_open_meteo_units,
    fetch_open_meteo_cluster_weather,
    fetch_open_meteo_point_weather,
    load_dwd,
)


class _OpenMeteoResponse:
    status_code = 200
    headers: dict[str, str] = {}
    url = "https://single-runs-api.open-meteo.com/v1/forecast"

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return {
            "hourly": {
                "time": ["2026-04-05T00:00"],
                "temperature_2m": [12.0],
            }
        }


class _OpenMeteoUnavailableResponse:
    status_code = 400
    headers: dict[str, str] = {}
    url = "https://customer-single-runs-api.open-meteo.com/v1/forecast"
    text = "Unexpected error while streaming data: modelRunUnavailable(model: icon_d2, run: 2025-10-24T06:00)"

    def raise_for_status(self) -> None:
        raise requests.HTTPError("400 Client Error: Bad Request", response=self)

    def json(self) -> dict:
        return {"error": True, "reason": self.text}


def test_fetch_open_meteo_batch_retries_connection_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"count": 0}

    def fake_get(*args, **kwargs):  # noqa: ANN002, ANN003, ARG001
        calls["count"] += 1
        if calls["count"] == 1:
            raise requests.exceptions.ConnectTimeout("timed out")
        return _OpenMeteoResponse()

    monkeypatch.setattr("da_price_forecasting.data.weather.requests.get", fake_get)
    monkeypatch.setattr("da_price_forecasting.data.weather.time.sleep", lambda seconds: None)

    items = _fetch_open_meteo_batch(
        base_url="https://single-runs-api.open-meteo.com/v1/forecast",
        latitude=[52.0],
        longitude=[13.0],
        hourly_variables=["temperature_2m"],
        model="icon_d2",
        cell_selection="nearest",
        timeout_seconds=60,
        start_date=date(2026, 4, 5),
        end_date=date(2026, 4, 5),
        retry_attempts=1,
        retry_backoff_seconds=0.0,
    )

    assert calls["count"] == 2
    assert items == [_OpenMeteoResponse().json()]


def test_fetch_open_meteo_batch_maps_400_model_run_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "da_price_forecasting.data.weather.requests.get",
        lambda *args, **kwargs: _OpenMeteoUnavailableResponse(),  # noqa: ARG005
    )

    with pytest.raises(OpenMeteoModelRunUnavailable):
        _fetch_open_meteo_batch(
            base_url="https://customer-single-runs-api.open-meteo.com/v1/forecast",
            latitude=[52.0],
            longitude=[13.0],
            hourly_variables=["temperature_2m"],
            model="icon_d2",
            cell_selection="nearest",
            timeout_seconds=60,
            run="2025-10-24T06:00",
            forecast_days=2,
            retry_attempts=0,
        )


def test_fetch_open_meteo_batch_falls_back_from_customer_endpoint_without_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = {}

    def fake_get(url, **kwargs):  # noqa: ANN001, ANN003
        request["url"] = url
        request["params"] = kwargs["params"]
        return _OpenMeteoResponse()

    monkeypatch.delenv("OPEN_METEO_API_KEY", raising=False)
    monkeypatch.setattr("da_price_forecasting.data.weather.requests.get", fake_get)

    items = _fetch_open_meteo_batch(
        base_url="https://customer-single-runs-api.open-meteo.com/v1/forecast",
        latitude=[52.0],
        longitude=[13.0],
        hourly_variables=["temperature_2m"],
        model="icon_d2",
        cell_selection="nearest",
        timeout_seconds=60,
        start_date=date(2026, 4, 5),
        end_date=date(2026, 4, 5),
        retry_attempts=0,
        api_key_env="OPEN_METEO_API_KEY",
    )

    assert request["url"] == "https://single-runs-api.open-meteo.com/v1/forecast"
    assert "apikey" not in request["params"]
    assert items == [_OpenMeteoResponse().json()]


def test_fetch_open_meteo_batch_raises_after_timeout_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_get(*args, **kwargs):  # noqa: ANN002, ANN003, ARG001
        raise requests.exceptions.ConnectTimeout("timed out")

    monkeypatch.setattr("da_price_forecasting.data.weather.requests.get", fake_get)
    monkeypatch.setattr("da_price_forecasting.data.weather.time.sleep", lambda seconds: None)

    with pytest.raises(requests.exceptions.ConnectTimeout):
        _fetch_open_meteo_batch(
            base_url="https://single-runs-api.open-meteo.com/v1/forecast",
            latitude=[52.0],
            longitude=[13.0],
            hourly_variables=["temperature_2m"],
            model="icon_d2",
            cell_selection="nearest",
            timeout_seconds=60,
            start_date=date(2026, 4, 5),
            end_date=date(2026, 4, 5),
            retry_attempts=1,
            retry_backoff_seconds=0.0,
        )


def test_open_meteo_normalisation_keeps_richer_wind_and_solar_fields() -> None:
    raw = pd.DataFrame(
        {
            "wind_speed_80m": [6.0],
            "wind_direction_80m": [270.0],
            "wind_speed_100m": [7.0],
            "wind_direction_100m": [270.0],
            "wind_speed_120m": [8.0],
            "wind_direction_120m": [270.0],
            "wind_gusts_10m": [10.0],
            "temperature_80m": [5.0],
            "temperature_120m": [4.0],
            "pressure_msl": [1012.0],
            "surface_pressure": [1000.0],
            "relative_humidity_2m": [80.0],
            "vapour_pressure_deficit": [0.4],
            "direct_normal_irradiance": [700.0],
            "sunshine_duration": [1800.0],
        }
    )

    normalised = _normalise_open_meteo_units(raw)

    for column in [
        "u80",
        "v80",
        "u100",
        "v100",
        "u120",
        "v120",
        "vmax10m",
        "t80m",
        "t120m",
        "msl",
        "sp",
        "r2m",
        "vpd",
        "dni",
        "sunshine_duration",
    ]:
        assert column in normalised.columns


def test_load_dwd_accepts_csvs_in_requested_run_folder_when_file_run_differs(tmp_path: Path) -> None:
    icon_dir = tmp_path / "icon"
    folder = icon_dir / "dwd_icon_daily_20251025_09"
    folder.mkdir(parents=True)
    (folder / "t2m_K_2025102508_raw.csv").write_text(
        "# Variable: t2m\n"
        "timestamp,cluster_0\n"
        "2025-10-25T00:00:00Z,280.0\n"
    )
    (folder / "ASWDIR_S_W_m-2_2025102507_instantaneous.csv").write_text(
        "# Variable: ASWDIR_S\n"
        "timestamp,cluster_0\n"
        "2025-10-25T00:15:00Z,100.0\n"
    )

    hourly, qh = load_dwd(
        icon_dir=icon_dir,
        start_folder_date=date(2025, 10, 25),
        required_run="09",
        target_tz="UTC",
    )

    assert "t2m_cluster_0" in hourly.columns
    assert "ASWDIR_cluster_0" in qh.columns
    assert hourly.loc[pd.Timestamp("2025-10-25T00:00:00Z"), "t2m_cluster_0"] == 280.0
    assert qh.loc[pd.Timestamp("2025-10-25T00:00:00Z"), "ASWDIR_cluster_0"] == 100.0


def test_single_run_backfill_preserves_existing_cache_days(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cluster_file = tmp_path / "clusters.csv"
    cluster_file.write_text("cluster_id,lat,lon\n0,52.0,13.0\n")
    cache_file = tmp_path / "open_meteo.csv"
    existing = pd.DataFrame(
        {"t2m_cluster_0": [283.15]},
        index=pd.DatetimeIndex([pd.Timestamp("2026-03-20T00:00:00+01:00")], name="timestamp"),
    )
    existing.to_csv(cache_file)

    def fake_fetch_open_meteo_batch(**kwargs):  # noqa: ANN003, ARG001
        return [
            {
                "hourly": {
                    "time": ["2026-03-20T23:00"],
                    "temperature_2m": [11.0],
                }
            }
        ]

    monkeypatch.setattr(weather, "_fetch_open_meteo_batch", fake_fetch_open_meteo_batch)

    fetch_open_meteo_cluster_weather(
        cluster_file=cluster_file,
        start_date=date(2026, 3, 21),
        end_date=date(2026, 3, 21),
        output_file=cache_file,
        hourly_variables=["temperature_2m"],
        batch_size=1,
        target_tz="Europe/Berlin",
        api_mode="single_run",
        single_run_hour_utc="06:00",
    )

    cached = pd.read_csv(cache_file, index_col=0)
    cached.index = pd.to_datetime(cached.index, utc=True).tz_convert("Europe/Berlin")
    cached_days = {timestamp.date().isoformat() for timestamp in cached.index.normalize()}

    assert cached_days == {"2026-03-20", "2026-03-21"}


def test_single_run_point_weather_keeps_point_level_columns(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    points = pd.DataFrame(
        {
            "weather_point_id": [0, 1],
            "lat": [52.0, 53.0],
            "lon": [8.0, 9.0],
        }
    )
    cache_file = tmp_path / "open_meteo_points.csv"

    def fake_fetch_open_meteo_batch(**kwargs):  # noqa: ANN003, ARG001
        return [
            {
                "hourly": {
                    "time": ["2026-03-20T23:00"],
                    "temperature_2m": [10.0],
                    "diffuse_radiation": [12.0],
                }
            },
            {
                "hourly": {
                    "time": ["2026-03-20T23:00"],
                    "temperature_2m": [20.0],
                    "diffuse_radiation": [24.0],
                }
            },
        ]

    monkeypatch.setattr(weather, "_fetch_open_meteo_batch", fake_fetch_open_meteo_batch)

    result = fetch_open_meteo_point_weather(
        points=points,
        start_date=date(2026, 3, 21),
        end_date=date(2026, 3, 21),
        output_file=cache_file,
        hourly_variables=["temperature_2m", "diffuse_radiation"],
        batch_size=2,
        target_tz="Europe/Berlin",
        api_mode="single_run",
        single_run_hour_utc="06:00",
    )

    assert "t2m_point_0" in result.columns
    assert "t2m_point_1" in result.columns
    assert "diffuse_point_0" in result.columns
    assert result["t2m_point_0"].iloc[0] == 283.15
    assert result["t2m_point_1"].iloc[0] == 293.15
    assert cache_file.exists()
