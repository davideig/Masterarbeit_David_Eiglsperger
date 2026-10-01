from __future__ import annotations

import json
from datetime import date, datetime, time
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, TypeVar
from zoneinfo import ZoneInfo

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..paths import find_repo_root

T = TypeVar("T", bound=BaseModel)


class WeatherSource(str, Enum):
    ERA5 = "ERA5"
    DWD = "DWD"


class ForecastVariant(str, Enum):
    FUNDAMENTAL = "fundamental"
    EXAA = "exaa"
    EXAA_ONLY = "exaa_only"


def _parse_timestamp(value: Any, target_tz: str) -> datetime:
    tz = ZoneInfo(target_tz)
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, date):
        dt = datetime.combine(value, time.min)
    elif hasattr(value, "to_pydatetime"):
        dt = value.to_pydatetime()
    else:
        raw = str(value)
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        dt = datetime.fromisoformat(raw)

    if dt.tzinfo is None:
        return dt.replace(tzinfo=tz)
    return dt.astimezone(tz)


def _default_datetime(value: str, target_tz: str = "Europe/Berlin") -> datetime:
    return _parse_timestamp(value, target_tz)


class RepoConfigModel(BaseModel):
    """Base model with repository-root-aware path handling."""

    model_config = ConfigDict(arbitrary_types_allowed=True, use_enum_values=False)

    repo_root: Path = Field(default_factory=find_repo_root)

    @field_validator("repo_root", mode="before")
    @classmethod
    def _coerce_repo_root(cls, value: Any) -> Path:
        return Path(value) if value is not None else find_repo_root()


def load_config_payload(path: Path) -> dict[str, Any]:
    """Load a JSON or YAML config file into a plain mapping."""
    suffix = path.suffix.lower()
    with open(path, encoding="utf-8") as handle:
        if suffix == ".json":
            payload = json.load(handle)
        elif suffix in {".yaml", ".yml"}:
            try:
                import yaml
            except ImportError as exc:
                raise ImportError("YAML configs require PyYAML. Install the project dependencies with Pixi.") from exc
            payload = yaml.safe_load(handle)
        else:
            raise ValueError(f"Unsupported config format for {path}. Use .json, .yaml, or .yml.")

    if payload is None:
        payload = {}
    if not isinstance(payload, dict):
        raise TypeError(f"Config {path} must contain a mapping at the top level.")
    return payload


def validate_config_payload(
    payload: Mapping[str, Any],
    model_cls: type[T],
    repo_root: Path | None = None,
) -> T:
    """Validate an already-loaded config mapping into a typed Pydantic model."""
    data = dict(payload)
    if "repo_root" not in data and repo_root is not None:
        data["repo_root"] = str(repo_root)
    return model_cls.model_validate(data)


def load_config(path: Path, model_cls: type[T]) -> T:
    """Load a JSON or YAML config file into a typed Pydantic model."""
    payload = load_config_payload(path)
    repo_root = find_repo_root(path.parent)
    return validate_config_payload(payload, model_cls, repo_root=repo_root)
