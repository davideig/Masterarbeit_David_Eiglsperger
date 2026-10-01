from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd

from ..config import MastrCapacityConfig


@dataclass(frozen=True)
class _TechnologySpec:
    label: str
    item_tag: str
    files_glob: str


_TECHNOLOGIES = [
    _TechnologySpec(label="pv", item_tag="EinheitSolar", files_glob="EinheitenSolar*.xml"),
    _TechnologySpec(label="wind", item_tag="EinheitWind", files_glob="EinheitenWind*.xml"),
]


def _text(element: ET.Element, tag: str) -> str | None:
    child = element.find(tag)
    if child is None or child.text is None:
        return None
    value = child.text.strip()
    return value or None


def _parse_float(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value.replace(",", "."))
    except ValueError:
        return None


def _parse_date(value: str | None) -> date | None:
    if value is None:
        return None
    try:
        return date.fromisoformat(value[:10])
    except ValueError:
        return None


def _capacity_mw(value: float | None, config: MastrCapacityConfig) -> float | None:
    if value is None:
        return None
    if config.capacity_unit != "kW":
        raise ValueError(f"Unsupported MaStR capacity unit: {config.capacity_unit!r}")
    return value / 1000.0


def _swept_area_m2(rotor_diameter_m: float | None) -> float | None:
    if rotor_diameter_m is None or rotor_diameter_m <= 0:
        return None
    radius_m = rotor_diameter_m / 2.0
    return 3.141592653589793 * radius_m**2


def _is_offshore(element: ET.Element) -> bool:
    wind_location = _text(element, "WindAnLandOderAufSee")
    if wind_location is None:
        return False
    return wind_location == "889"


def _iter_units(file: Path, item_tag: str) -> Iterable[ET.Element]:
    for _event, element in ET.iterparse(file, events=("end",)):
        if element.tag == item_tag:
            yield element
            element.clear()


def _unit_to_row(element: ET.Element, technology: str, config: MastrCapacityConfig) -> dict[str, object] | None:
    if _text(element, "Land") != config.country_code:
        return None

    commissioning_date = _parse_date(_text(element, "Inbetriebnahmedatum"))
    decommissioning_date = _parse_date(_text(element, "DatumEndgueltigeStilllegung"))
    operating_status = _text(element, "EinheitBetriebsstatus")
    system_status = _text(element, "EinheitSystemstatus")

    if config.active_only:
        if operating_status != config.active_status_code:
            return None
        if decommissioning_date is not None:
            return None

    if config.max_commissioning_date is not None:
        if commissioning_date is None or commissioning_date > config.max_commissioning_date:
            return None

    capacity_tag = "Nettonennleistung" if config.capacity_source == "net" else "Bruttoleistung"
    capacity_mw = _capacity_mw(_parse_float(_text(element, capacity_tag)), config)
    lat = _parse_float(_text(element, "Breitengrad"))
    lon = _parse_float(_text(element, "Laengengrad"))

    if capacity_mw is None or capacity_mw <= 0 or lat is None or lon is None:
        return None
    if not (47.0 <= lat <= 56.5 and 5.0 <= lon <= 16.5):
        return None

    if technology == "wind":
        technology = "wind_offshore" if _is_offshore(element) else "wind_onshore"

    row = {
        "technology": technology,
        "capacity_mw": capacity_mw,
        "lat": lat,
        "lon": lon,
        "commissioning_date": commissioning_date.isoformat() if commissioning_date else None,
        "decommissioning_date": decommissioning_date.isoformat() if decommissioning_date else None,
        "operating_status": operating_status,
        "system_status": system_status,
        "mastr_number": _text(element, "EinheitMastrNummer"),
        "federal_state_code": _text(element, "Bundesland"),
        "municipality_code": _text(element, "Gemeindeschluessel"),
    }

    if config.include_wind_turbine_metadata and technology.startswith("wind_"):
        hub_height_m = _parse_float(_text(element, "Nabenhoehe"))
        rotor_diameter_m = _parse_float(_text(element, "Rotordurchmesser"))
        swept_area_m2 = _swept_area_m2(rotor_diameter_m)
        row.update(
            {
                "wind_location_code": _text(element, "WindAnLandOderAufSee"),
                "manufacturer_code": _text(element, "Hersteller"),
                "turbine_type": _text(element, "Typenbezeichnung"),
                "wind_park_name": _text(element, "NameWindpark"),
                "hub_height_m": hub_height_m,
                "rotor_diameter_m": rotor_diameter_m,
                "swept_area_m2": swept_area_m2,
                "specific_power_mw_per_m2": (
                    capacity_mw / swept_area_m2 if swept_area_m2 is not None and swept_area_m2 > 0 else None
                ),
                "rotor_blade_deicing": _text(element, "Rotorblattenteisungssystem"),
                "curtailment_limitation_flag": _text(element, "AuflageAbschaltungLeistungsbegrenzung"),
                "remote_control_grid_operator": _text(element, "FernsteuerbarkeitNb"),
                "remote_control_direct_marketer": _text(element, "FernsteuerbarkeitDv"),
            }
        )

    return row


def build_mastr_capacity(config: MastrCapacityConfig) -> pd.DataFrame:
    if not config.mastr_dir.exists():
        raise FileNotFoundError(f"MaStR export directory does not exist: {config.mastr_dir}")

    rows: list[dict[str, object]] = []
    stats: dict[str, int] = {}

    for spec in _TECHNOLOGIES:
        files = sorted(config.mastr_dir.glob(spec.files_glob))
        if not files:
            raise FileNotFoundError(f"No {spec.files_glob} files found in {config.mastr_dir}")

        kept = 0
        parsed = 0
        for file in files:
            for element in _iter_units(file, spec.item_tag):
                parsed += 1
                row = _unit_to_row(element, spec.label, config)
                if row is not None:
                    rows.append(row)
                    kept += 1
        stats[f"{spec.label}_parsed"] = parsed
        stats[f"{spec.label}_kept"] = kept

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("No wind/PV capacity rows survived the MaStR filters.")

    df = df.sort_values(["technology", "commissioning_date", "mastr_number"], na_position="last").reset_index(drop=True)
    config.output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.output_file, index=False)

    metadata = {
        "mastr_dir": str(config.mastr_dir),
        "output_file": str(config.output_file),
        "active_only": config.active_only,
        "max_commissioning_date": config.max_commissioning_date.isoformat()
        if config.max_commissioning_date is not None
        else None,
        "capacity_source": config.capacity_source,
        "capacity_unit": config.capacity_unit,
        "rows": len(df),
        "capacity_mw_by_technology": df.groupby("technology")["capacity_mw"].sum().to_dict(),
        "stats": stats,
    }
    config.output_file.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"[MaStR] Saved {len(df):,} renewable units to {config.output_file}")
    print("[MaStR] Capacity by technology [MW]:")
    for technology, capacity in metadata["capacity_mw_by_technology"].items():
        print(f"  {technology}: {capacity:,.1f}")

    return df


def run_mastr_capacity(config: MastrCapacityConfig) -> pd.DataFrame:
    return build_mastr_capacity(config)
