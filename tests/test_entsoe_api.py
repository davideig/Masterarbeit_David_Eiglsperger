from __future__ import annotations

import importlib
import sys
import types

import pandas as pd


def _load_entsoe_module(monkeypatch, pandas_client_cls, raw_client_cls, parse_prices):
    entsoe_module = types.ModuleType("entsoe")
    entsoe_module.EntsoePandasClient = pandas_client_cls
    entsoe_module.EntsoeRawClient = raw_client_cls

    parsers_module = types.ModuleType("entsoe.parsers")
    parsers_module.parse_prices = parse_prices

    monkeypatch.setitem(sys.modules, "entsoe", entsoe_module)
    monkeypatch.setitem(sys.modules, "entsoe.parsers", parsers_module)
    sys.modules.pop("da_price_forecasting.data.entsoe", None)
    return importlib.import_module("da_price_forecasting.data.entsoe")


def _hourly_series(start: pd.Timestamp, name: str) -> pd.Series:
    index = pd.date_range(start=start, periods=24, freq="h")
    return pd.Series(range(24), index=index, dtype=float, name=name)


def test_fetch_prices_queries_short_window_and_expands_to_quarter_hours(monkeypatch) -> None:
    class FakePandasClient:
        instances = []

        def __init__(self, api_key: str) -> None:
            self.api_key = api_key
            self.requests = []
            self.instances.append(self)

        def query_day_ahead_prices(self, country_code, start, end):
            self.requests.append((country_code, start, end))
            return _hourly_series(start, "price_da")

    class FakeRawClient:
        pass

    monkeypatch.setenv("ENTSOE_TEST_API_KEY", "secret")
    entsoe = _load_entsoe_module(monkeypatch, FakePandasClient, FakeRawClient, lambda xml: None)

    start_day = pd.Timestamp("2026-03-01", tz="Europe/Berlin")
    result = entsoe.fetch_prices(
        start_day=start_day,
        end_day=start_day,
        api_key_env="ENTSOE_TEST_API_KEY",
    )

    assert len(result) == 96
    assert result.index[0] == pd.Timestamp("2026-03-01T00:00:00+01:00")
    assert result.index[-1] == pd.Timestamp("2026-03-01T23:45:00+01:00")
    assert result["price_da"].iloc[:4].tolist() == [0.0, 0.0, 0.0, 0.0]
    assert result["price_da"].iloc[4] == 1.0

    client = FakePandasClient.instances[0]
    assert client.api_key == "secret"
    country_code, query_start, query_end = client.requests[0]
    assert country_code == "DE_LU"
    assert query_start == pd.Timestamp("2026-03-01T00:00:00+01:00")
    assert query_end == pd.Timestamp("2026-03-02T00:00:00+01:00")


def test_fetch_foreign_day_ahead_prices_uses_chunks(monkeypatch) -> None:
    class FakePandasClient:
        instances = []

        def __init__(self, api_key: str) -> None:
            self.api_key = api_key
            self.requests = []
            self.instances.append(self)

        def query_day_ahead_prices(self, country_code, start, end):
            self.requests.append((country_code, start, end))
            return _hourly_series(start, "price_da")

    class FakeRawClient:
        pass

    monkeypatch.setenv("ENTSOE_TEST_API_KEY", "secret")
    entsoe = _load_entsoe_module(monkeypatch, FakePandasClient, FakeRawClient, lambda xml: None)

    start_day = pd.Timestamp("2026-03-01", tz="Europe/Berlin")
    result = entsoe.fetch_foreign_day_ahead_prices(
        start_day=start_day,
        end_day=start_day + pd.Timedelta(days=1),
        markets=["CH"],
        api_key_env="ENTSOE_TEST_API_KEY",
        chunk_days=1,
    )

    assert len(result) == 2 * 96
    assert result.columns.tolist() == ["price_da_CH"]

    client = FakePandasClient.instances[0]
    assert client.requests == [
        (
            "CH",
            pd.Timestamp("2026-03-01T00:00:00+01:00"),
            pd.Timestamp("2026-03-02T00:00:00+01:00"),
        ),
        (
            "CH",
            pd.Timestamp("2026-03-02T00:00:00+01:00"),
            pd.Timestamp("2026-03-03T00:00:00+01:00"),
        ),
    ]


def test_fetch_load_forecast_accepts_dataframe_response(monkeypatch) -> None:
    class FakePandasClient:
        def __init__(self, api_key: str) -> None:
            self.api_key = api_key

        def query_load_forecast(self, country_code, start, end):
            return _hourly_series(start, "load").to_frame()

    class FakeRawClient:
        pass

    monkeypatch.setenv("ENTSOE_TEST_API_KEY", "secret")
    entsoe = _load_entsoe_module(monkeypatch, FakePandasClient, FakeRawClient, lambda xml: None)

    start_day = pd.Timestamp("2026-03-01", tz="Europe/Berlin")
    result = entsoe.fetch_load_forecast(
        start_day=start_day,
        end_day=start_day,
        api_key_env="ENTSOE_TEST_API_KEY",
    )

    assert len(result) == 96
    assert result.columns.tolist() == ["load_fc"]
    assert result["load_fc"].iloc[0] == 0.0
    assert result["load_fc"].iloc[-1] == 23.0


def test_fetch_actual_load_prefers_actual_column(monkeypatch) -> None:
    class FakePandasClient:
        def __init__(self, api_key: str) -> None:
            self.api_key = api_key

        def query_load(self, country_code, start, end):
            forecast = _hourly_series(start, "Forecasted Load")
            actual = (_hourly_series(start, "Actual Load") + 100.0).rename("Actual Load")
            return pd.concat([forecast, actual], axis=1)

    class FakeRawClient:
        pass

    monkeypatch.setenv("ENTSOE_TEST_API_KEY", "secret")
    entsoe = _load_entsoe_module(monkeypatch, FakePandasClient, FakeRawClient, lambda xml: None)

    start_day = pd.Timestamp("2026-03-01", tz="Europe/Berlin")
    result = entsoe.fetch_actual_load(
        start_day=start_day,
        end_day=start_day,
        api_key_env="ENTSOE_TEST_API_KEY",
    )

    assert len(result) == 96
    assert result.columns.tolist() == ["load_actual"]
    assert result["load_actual"].iloc[0] == 100.0
    assert result["load_actual"].iloc[-1] == 123.0


def test_fetch_actual_solar_generation_by_control_area_queries_each_area(monkeypatch) -> None:
    class FakePandasClient:
        instances = []

        def __init__(self, api_key: str) -> None:
            self.api_key = api_key
            self.requests = []
            self.instances.append(self)

        def query_generation(self, country_code, start, end, psr_type, nett):
            self.requests.append((country_code, start, end, psr_type, nett))
            return (_hourly_series(start, "generation") + len(self.requests) * 100.0).to_frame()

    class FakeRawClient:
        pass

    monkeypatch.setenv("ENTSOE_TEST_API_KEY", "secret")
    entsoe = _load_entsoe_module(monkeypatch, FakePandasClient, FakeRawClient, lambda xml: None)

    start_day = pd.Timestamp("2026-03-01", tz="Europe/Berlin")
    result = entsoe.fetch_actual_solar_generation_by_control_area(
        start_day=start_day,
        end_day=start_day,
        control_area_targets={
            "Solar_50Hertz_Actual_MW": "10YDE-VE-------2",
            "Solar_Amprion_Actual_MW": "10YDE-RWENET---I",
        },
        api_key_env="ENTSOE_TEST_API_KEY",
    )

    assert len(result) == 96
    assert result.columns.tolist() == [
        "Solar_50Hertz_Actual_MW",
        "Solar_Amprion_Actual_MW",
        "Solar_Control_Area_Total_Actual_MW",
    ]
    assert result["Solar_50Hertz_Actual_MW"].iloc[:4].tolist() == [100.0] * 4
    assert result["Solar_Amprion_Actual_MW"].iloc[:4].tolist() == [200.0] * 4
    assert result["Solar_Control_Area_Total_Actual_MW"].iloc[:4].tolist() == [300.0] * 4

    client = FakePandasClient.instances[0]
    assert client.requests == [
        (
            "10YDE-VE-------2",
            pd.Timestamp("2026-03-01T00:00:00+01:00"),
            pd.Timestamp("2026-03-02T00:00:00+01:00"),
            "B16",
            False,
        ),
        (
            "10YDE-RWENET---I",
            pd.Timestamp("2026-03-01T00:00:00+01:00"),
            pd.Timestamp("2026-03-02T00:00:00+01:00"),
            "B16",
            False,
        ),
    ]


def test_fetch_ntc_data_aggregates_bidirectional_neighbor_capacity(monkeypatch) -> None:
    class FakePandasClient:
        instances = []

        def __init__(self, api_key: str) -> None:
            self.api_key = api_key
            self.requests = []
            self.instances.append(self)

        def query_net_transfer_capacity_dayahead(self, country_code_from, country_code_to, start, end):
            self.requests.append((country_code_from, country_code_to, start, end))
            index = pd.date_range(start=start, periods=24, freq="h")

            if (country_code_from, country_code_to) == ("FR", "DE_LU"):
                return pd.Series(100.0, index=index, name="fr_import")
            if (country_code_from, country_code_to) == ("DE_LU", "FR"):
                return pd.Series(30.0, index=index, name="fr_export").to_frame()
            if (country_code_from, country_code_to) == ("NL", "DE_LU"):
                raise RuntimeError("missing border data")
            if (country_code_from, country_code_to) == ("DE_LU", "NL"):
                return pd.Series(5.0, index=index, name="nl_export")
            raise AssertionError((country_code_from, country_code_to))

    class FakeRawClient:
        pass

    monkeypatch.setenv("ENTSOE_TEST_API_KEY", "secret")
    entsoe = _load_entsoe_module(monkeypatch, FakePandasClient, FakeRawClient, lambda xml: None)

    start_day = pd.Timestamp("2026-03-01", tz="Europe/Berlin")
    result = entsoe.fetch_ntc_data(
        start_day=start_day,
        end_day=start_day,
        neighbors=["FR", "NL"],
        api_key_env="ENTSOE_TEST_API_KEY",
    )

    assert len(result) == 96
    assert result.columns.tolist() == ["NTC_Net_MW"]
    assert result.index[0] == pd.Timestamp("2026-03-01T00:00:00+01:00")
    assert result.index[-1] == pd.Timestamp("2026-03-01T23:45:00+01:00")
    assert result["NTC_Net_MW"].iloc[:4].tolist() == [65.0, 65.0, 65.0, 65.0]
    assert result["NTC_Net_MW"].iloc[-1] == 65.0

    client = FakePandasClient.instances[0]
    assert client.api_key == "secret"
    assert client.requests == [
        (
            "FR",
            "DE_LU",
            pd.Timestamp("2026-03-01T00:00:00+01:00"),
            pd.Timestamp("2026-03-02T00:00:00+01:00"),
        ),
        (
            "DE_LU",
            "FR",
            pd.Timestamp("2026-03-01T00:00:00+01:00"),
            pd.Timestamp("2026-03-02T00:00:00+01:00"),
        ),
        (
            "NL",
            "DE_LU",
            pd.Timestamp("2026-03-01T00:00:00+01:00"),
            pd.Timestamp("2026-03-02T00:00:00+01:00"),
        ),
        (
            "DE_LU",
            "NL",
            pd.Timestamp("2026-03-01T00:00:00+01:00"),
            pd.Timestamp("2026-03-02T00:00:00+01:00"),
        ),
    ]


def test_fetch_exaa_prices_parses_raw_api_xml(monkeypatch) -> None:
    class FakePandasClient:
        pass

    class FakeRawClient:
        def __init__(self, api_key: str) -> None:
            self.api_key = api_key

        def query_day_ahead_prices(self, country_code, start, end, sequence):
            assert country_code == "DE_LU"
            assert sequence == 2
            return "<publication-market-document/>"

    def fake_parse_prices(xml):
        assert xml == "<publication-market-document/>"
        return _hourly_series(pd.Timestamp("2026-03-01T00:00:00+01:00"), "price_exaa")

    monkeypatch.setenv("ENTSOE_TEST_API_KEY", "secret")
    entsoe = _load_entsoe_module(monkeypatch, FakePandasClient, FakeRawClient, fake_parse_prices)

    start_day = pd.Timestamp("2026-03-01", tz="Europe/Berlin")
    result = entsoe.fetch_prices_exaa(
        start_day=start_day,
        end_day=start_day,
        api_key_env="ENTSOE_TEST_API_KEY",
    )

    assert len(result) == 96
    assert result.columns.tolist() == ["price_exaa"]
    assert result["price_exaa"].iloc[0] == 0.0
    assert result["price_exaa"].iloc[-1] == 23.0


def test_fetch_exaa_prices_combines_multiple_parsed_series(monkeypatch) -> None:
    class FakePandasClient:
        pass

    class FakeRawClient:
        def __init__(self, api_key: str) -> None:
            self.api_key = api_key

        def query_day_ahead_prices(self, country_code, start, end, sequence):
            return "<publication-market-document/>"

    def fake_parse_prices(xml):
        return {
            "day_1": _hourly_series(pd.Timestamp("2026-03-01T00:00:00+01:00"), "price_exaa"),
            "day_2": _hourly_series(pd.Timestamp("2026-03-02T00:00:00+01:00"), "price_exaa") + 100,
        }

    monkeypatch.setenv("ENTSOE_TEST_API_KEY", "secret")
    entsoe = _load_entsoe_module(monkeypatch, FakePandasClient, FakeRawClient, fake_parse_prices)

    result = entsoe.fetch_prices_exaa(
        start_day=pd.Timestamp("2026-03-01", tz="Europe/Berlin"),
        end_day=pd.Timestamp("2026-03-02", tz="Europe/Berlin"),
        api_key_env="ENTSOE_TEST_API_KEY",
    )

    assert len(result) == 192
    assert result.loc[pd.Timestamp("2026-03-01T00:00:00+01:00"), "price_exaa"] == 0.0
    assert result.loc[pd.Timestamp("2026-03-02T00:00:00+01:00"), "price_exaa"] == 100.0


def test_fetch_generation_unavailability_aggregates_overlapping_outages(monkeypatch) -> None:
    class FakePandasClient:
        instances = []

        def __init__(self, api_key: str) -> None:
            self.api_key = api_key
            self.requests = []
            self.instances.append(self)

        def query_unavailability_of_generation_units(self, country_code, start, end):
            self.requests.append((country_code, start, end))
            return pd.DataFrame(
                {
                    "start": [
                        pd.Timestamp("2026-03-01T00:00:00+01:00"),
                        pd.Timestamp("2026-03-01T00:30:00+01:00"),
                        pd.Timestamp("2026-03-01T00:00:00+01:00"),
                        pd.Timestamp("2026-03-01T01:00:00+01:00"),
                    ],
                    "end": [
                        pd.Timestamp("2026-03-01T01:00:00+01:00"),
                        pd.Timestamp("2026-03-01T01:15:00+01:00"),
                        pd.Timestamp("2026-03-01T00:30:00+01:00"),
                        pd.Timestamp("2026-03-01T01:30:00+01:00"),
                    ],
                    "nominal_power": ["100", "80", "50", "30"],
                    "avail_qty": ["40", "10", "0", "30"],
                    "docstatus": ["Active", "Active", "Cancelled", "Active"],
                }
            )

    class FakeRawClient:
        pass

    monkeypatch.setenv("ENTSOE_TEST_API_KEY", "secret")
    entsoe = _load_entsoe_module(monkeypatch, FakePandasClient, FakeRawClient, lambda xml: None)

    result = entsoe.fetch_generation_unavailability(
        start_day=pd.Timestamp("2026-03-01", tz="Europe/Berlin"),
        end_day=pd.Timestamp("2026-03-01", tz="Europe/Berlin"),
        api_key_env="ENTSOE_TEST_API_KEY",
    )

    assert result.columns.tolist() == ["Unavail_Total_MW"]
    assert len(result) == 96
    assert result.index[0] == pd.Timestamp("2026-03-01T00:00:00+01:00")
    assert result.index[-1] == pd.Timestamp("2026-03-01T23:45:00+01:00")
    assert result["Unavail_Total_MW"].iloc[:8].tolist() == [
        60.0,
        60.0,
        130.0,
        130.0,
        70.0,
        0.0,
        0.0,
        0.0,
    ]

    client = FakePandasClient.instances[0]
    assert client.api_key == "secret"
    assert client.requests[0] == (
        "DE_LU",
        pd.Timestamp("2026-03-01T00:00:00+01:00"),
        pd.Timestamp("2026-03-02T00:00:00+01:00"),
    )
