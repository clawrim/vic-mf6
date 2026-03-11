#!/usr/bin/env python3

###############################################################################
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2025 Abdullah Azzam and Huidae Cho
#
# License
# This source code is licensed under the GNU General Public License v3.0 or
# later (GPL-3.0-or-later). You may use, study, modify, and redistribute this
# code under the same license terms. Any derivative work must be released under
# GPL-compatible terms with source code disclosure. This software is provided
# “as is,” without warranty of any kind. See the COPYING file for details.
#
# Contact
# Abdullah Azzam <abdazzam@nmsu.edu>
# Department of Civil and Environmental Engineering,
# New Mexico State University
###############################################################################

from __future__ import annotations

"""load and validate the vic-mf6 coupling configuration.

this module keeps configuration concerns out of the runtime modules.
that separation matters for two reasons.

first, the runtime modules should be able to assume that paths, dates, and
optional knobs have already been validated. that keeps the scientific code
focused on hydrology and model control rather than repeated defensive parsing.

second, config loading is one of the first places where product quality shows.
a bad path or malformed date should fail immediately with a direct error
message, instead of surfacing later as a confusing bmi or netcdf error.
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


class ConfigurationError(RuntimeError):
    """raised when the yaml configuration is missing data or contains invalid values."""


@dataclass(frozen=True, slots=True)
class MF6RuntimeConfig:
    """settings required to initialize mf6 through bmi or xmipy."""

    workspace: str
    dll: str
    start_date: datetime
    length_units: str = "meters"


@dataclass(frozen=True, slots=True)
class VicImageDriverConfig:
    """settings required to run the vic image driver for each coupling step."""

    working_directory: str
    executable_path: str
    global_parameter_template: str
    outputs_directory: str
    exchange_directory: str
    parameters_netcdf_path: str
    water_balance_variable: str
    initial_moisture_layer_index: int
    spawn_timeout_seconds: int = 3600


@dataclass(frozen=True, slots=True)
class CouplingConfig:
    """settings that define how vic and mf6 exchange fluxes and time windows."""

    coupling_table_csv: str
    start_date: datetime
    end_date: datetime
    recharge_scale: float = 1.0
    vic_grid_shape: tuple[int, int] | None = None


@dataclass(frozen=True, slots=True)
class ApplicationConfig:
    """top-level configuration object used by the coupling entry points."""

    mf6: MF6RuntimeConfig
    vic: VicImageDriverConfig
    coupling: CouplingConfig
    config_path: str
    config_directory: str


# compatibility aliases kept inside the final module so downstream code can
# adopt the shorter mf6 naming without forcing an all-at-once migration.
Modflow6RuntimeConfig = MF6RuntimeConfig


def _application_config_modflow6_property(
    self: "ApplicationConfig",
) -> MF6RuntimeConfig:
    """compatibility view for older code that still expects config.modflow6."""

    return self.mf6


ApplicationConfig.modflow6 = property(_application_config_modflow6_property)


_REQUIRED_TOP_LEVEL_SECTIONS = ("mf6", "vic", "coupling")
_REQUIRED_MF6_KEYS = ("workspace", "dll", "start_date")
_REQUIRED_VIC_KEYS = (
    "dir",
    "exe",
    "global_param",
    "outputs_dir",
    "exchange_dir",
    "params_file",
    "wbal_var",
    "init_moist_layer",
)
_REQUIRED_COUPLING_KEYS = ("table_csv", "start_date", "end_date")


def load_application_config(config_path: str | Path) -> ApplicationConfig:
    """read, validate, and normalize a vic-mf6 yaml configuration file.

    path resolution follows two simple rules.

    1. top-level paths such as mf6.workspace, mf6.dll, vic.dir, vic.exe, and
       coupling.table_csv are resolved relative to the directory that contains
       the yaml file.
    2. vic subordinate paths that conceptually live inside the vic run tree
       (global_param, outputs_dir, exchange_dir, and params_file) are resolved
       relative to vic.dir unless they are already absolute.

    this matches the mental model most users have when they write the config.
    the yaml file remains portable across launch directories, while the vic
    subsection still behaves like a self-contained run description.
    """

    config_file = Path(config_path).expanduser().resolve()
    if not config_file.exists():
        raise ConfigurationError(f"configuration file was not found: {config_file}")
    if not config_file.is_file():
        raise ConfigurationError(f"configuration path is not a file: {config_file}")

    try:
        raw_config = yaml.safe_load(config_file.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        raise ConfigurationError(f"failed to parse yaml: {config_file}") from exc

    if not isinstance(raw_config, dict):
        raise ConfigurationError("top-level yaml content must be a mapping")

    _validate_required_sections(raw_config)
    _validate_required_keys(raw_config)

    config_directory = config_file.parent

    mf6_config = MF6RuntimeConfig(
        workspace=_resolve_config_path(
            config_directory, raw_config["mf6"]["workspace"]
        ),
        dll=_resolve_config_path(config_directory, raw_config["mf6"]["dll"]),
        start_date=_parse_datetime(
            raw_config["mf6"]["start_date"], field_name="mf6.start_date"
        ),
        length_units=str(raw_config["mf6"].get("length_units", "meters")).strip()
        or "meters",
    )

    vic_working_directory = _resolve_config_path(
        config_directory, raw_config["vic"]["dir"]
    )

    vic_config = VicImageDriverConfig(
        working_directory=vic_working_directory,
        executable_path=_resolve_config_path(
            config_directory, raw_config["vic"]["exe"]
        ),
        global_parameter_template=_resolve_path_relative_to_base_directory(
            vic_working_directory,
            raw_config["vic"]["global_param"],
            field_name="vic.global_param",
        ),
        outputs_directory=_resolve_path_relative_to_base_directory(
            vic_working_directory,
            raw_config["vic"]["outputs_dir"],
            field_name="vic.outputs_dir",
        ),
        exchange_directory=_resolve_path_relative_to_base_directory(
            vic_working_directory,
            raw_config["vic"]["exchange_dir"],
            field_name="vic.exchange_dir",
        ),
        parameters_netcdf_path=_resolve_path_relative_to_base_directory(
            vic_working_directory,
            raw_config["vic"]["params_file"],
            field_name="vic.params_file",
        ),
        water_balance_variable=str(raw_config["vic"]["wbal_var"]).strip(),
        initial_moisture_layer_index=_parse_int(
            raw_config["vic"]["init_moist_layer"], field_name="vic.init_moist_layer"
        ),
        spawn_timeout_seconds=_parse_positive_int(
            raw_config["vic"].get("spawn_timeout_seconds", 3600),
            field_name="vic.spawn_timeout_seconds",
        ),
    )

    coupling_config = CouplingConfig(
        coupling_table_csv=_resolve_config_path(
            config_directory, raw_config["coupling"]["table_csv"]
        ),
        start_date=_parse_datetime(
            raw_config["coupling"]["start_date"], field_name="coupling.start_date"
        ),
        end_date=_parse_datetime(
            raw_config["coupling"]["end_date"], field_name="coupling.end_date"
        ),
        recharge_scale=_parse_float(
            raw_config["coupling"].get("recharge_scale", 1.0),
            field_name="coupling.recharge_scale",
        ),
        vic_grid_shape=_parse_optional_shape(
            raw_config["coupling"].get("vic_grid_shape"),
            field_name="coupling.vic_grid_shape",
        ),
    )

    if coupling_config.end_date < coupling_config.start_date:
        raise ConfigurationError(
            "coupling.end_date must be greater than or equal to coupling.start_date"
        )

    return ApplicationConfig(
        mf6=mf6_config,
        vic=vic_config,
        coupling=coupling_config,
        config_path=str(config_file),
        config_directory=str(config_directory),
    )


# legacy alias kept for easier migration from the older code.
ConfigError = ConfigurationError


def load_config(config_path: str | Path) -> dict[str, Any]:
    """return a dictionary shaped like the legacy loader output.

    the refactored code prefers dataclasses because they make the contract clear.
    this function remains available so existing scripts can migrate gradually.
    """

    application_config = load_application_config(config_path)

    return {
        "mf6": {
            "workspace": application_config.mf6.workspace,
            "dll": application_config.mf6.dll,
            "start_date": application_config.mf6.start_date.isoformat(),
            "length_units": application_config.mf6.length_units,
        },
        "vic": {
            "dir": application_config.vic.working_directory,
            "exe": application_config.vic.executable_path,
            "global_param": application_config.vic.global_parameter_template,
            "outputs_dir": application_config.vic.outputs_directory,
            "exchange_dir": application_config.vic.exchange_directory,
            "params_file": application_config.vic.parameters_netcdf_path,
            "wbal_var": application_config.vic.water_balance_variable,
            "init_moist_layer": application_config.vic.initial_moisture_layer_index,
            "spawn_timeout_seconds": application_config.vic.spawn_timeout_seconds,
        },
        "coupling": {
            "table_csv": application_config.coupling.coupling_table_csv,
            "start_date": application_config.coupling.start_date.isoformat(),
            "end_date": application_config.coupling.end_date.isoformat(),
            "recharge_scale": application_config.coupling.recharge_scale,
            "vic_grid_shape": application_config.coupling.vic_grid_shape,
        },
    }


def _validate_required_sections(raw_config: dict[str, Any]) -> None:
    for section_name in _REQUIRED_TOP_LEVEL_SECTIONS:
        if section_name not in raw_config:
            raise ConfigurationError(f"missing configuration section: {section_name}")
        if not isinstance(raw_config[section_name], dict):
            raise ConfigurationError(
                f"configuration section must be a mapping: {section_name}"
            )


def _validate_required_keys(raw_config: dict[str, Any]) -> None:
    _validate_required_section_keys(
        raw_config["mf6"], _REQUIRED_MF6_KEYS, section_name="mf6"
    )
    _validate_required_section_keys(
        raw_config["vic"], _REQUIRED_VIC_KEYS, section_name="vic"
    )
    _validate_required_section_keys(
        raw_config["coupling"], _REQUIRED_COUPLING_KEYS, section_name="coupling"
    )


def _validate_required_section_keys(
    section_values: dict[str, Any],
    required_keys: tuple[str, ...],
    *,
    section_name: str,
) -> None:
    for key_name in required_keys:
        if key_name not in section_values:
            raise ConfigurationError(
                f"missing configuration value: {section_name}.{key_name}"
            )


def _resolve_config_path(config_directory: Path, raw_value: Any) -> str:
    if not isinstance(raw_value, str):
        raise ConfigurationError(
            f"expected a path string, got {type(raw_value).__name__}"
        )

    candidate = Path(raw_value).expanduser()
    if not candidate.is_absolute():
        candidate = config_directory / candidate

    return str(candidate.resolve())


def _resolve_path_relative_to_base_directory(
    base_directory: str | Path,
    raw_value: Any,
    *,
    field_name: str,
) -> str:
    """resolve a possibly-relative path against a declared base directory.

    the vic subsection in the yaml intentionally behaves like a small nested
    namespace. once vic.dir is known, paths that represent files or directories
    inside the vic run tree should resolve from there. without this rule, moving
    the yaml file or launching from a different shell directory would silently
    redirect vic to the wrong netcdf files or output folders.
    """

    if not isinstance(raw_value, str):
        raise ConfigurationError(f"{field_name} must be a path string")

    candidate = Path(raw_value).expanduser()
    if candidate.is_absolute():
        return str(candidate.resolve())

    return str((Path(base_directory) / candidate).resolve())


def _parse_datetime(raw_value: Any, *, field_name: str) -> datetime:
    if not isinstance(raw_value, str):
        raise ConfigurationError(f"{field_name} must be an iso date or datetime string")

    try:
        return datetime.fromisoformat(raw_value)
    except ValueError as exc:
        raise ConfigurationError(
            f"{field_name} is not a valid iso date or datetime string"
        ) from exc


def _parse_int(raw_value: Any, *, field_name: str) -> int:
    try:
        return int(raw_value)
    except Exception as exc:
        raise ConfigurationError(f"{field_name} must be an integer") from exc


def _parse_positive_int(raw_value: Any, *, field_name: str) -> int:
    value = _parse_int(raw_value, field_name=field_name)
    if value <= 0:
        raise ConfigurationError(f"{field_name} must be greater than zero")
    return value


def _parse_float(raw_value: Any, *, field_name: str) -> float:
    try:
        return float(raw_value)
    except Exception as exc:
        raise ConfigurationError(f"{field_name} must be numeric") from exc


def _parse_optional_shape(raw_value: Any, *, field_name: str) -> tuple[int, int] | None:
    if raw_value is None:
        return None
    if not isinstance(raw_value, (list, tuple)) or len(raw_value) != 2:
        raise ConfigurationError(f"{field_name} must contain exactly two integers")

    first = _parse_positive_int(raw_value[0], field_name=f"{field_name}[0]")
    second = _parse_positive_int(raw_value[1], field_name=f"{field_name}[1]")
    return (first, second)
