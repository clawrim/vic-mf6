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
#   Abdullah Azzam <abdazzam@nmsu.edu>
#   Department of Civil Engineering, New Mexico State University
###############################################################################
"""
Parse and validate YAML config for VIC–MF6 runs, including paths, dates, and logging options.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


class ConfigError(Exception):
    """bad or missing configuration"""


def load_config(path: str | os.PathLike[str]) -> dict[str, Any]:
    """load yaml config and validate required keys"""
    p = Path(path).expanduser()
    if not p.exists():
        raise ConfigError(f"missing config: {p}")

    try:
        with p.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    except Exception as e:
        raise ConfigError(f"failed to read yaml: {e}") from e

    _validate(cfg)
    _normalize_paths(cfg)
    return cfg


def _validate(cfg: dict[str, Any]) -> None:
    # top-level sections
    for sec in ("mf6", "vic", "coupling"):
        if sec not in cfg or not isinstance(cfg[sec], dict):
            raise ConfigError(f"missing section: {sec}")

    # mf6
    for key in ("workspace", "dll", "start_date", "model_name"):
        if key not in cfg["mf6"]:
            raise ConfigError(f"mf6.{key} is required")

    # vic
    for key in (
        "dir",
        "exe",
        "global_param",
        "outputs_dir",
        "exchange_dir",
        "params_file",
        "wbal_var",
        "init_moist_layer",
    ):
        if key not in cfg["vic"]:
            raise ConfigError(f"vic.{key} is required")

    # coupling
    for key in ("table_csv", "start_date", "end_date", "vic_grid_shape"):
        if key not in cfg["coupling"]:
            raise ConfigError(f"coupling.{key} is required")

    # types and shapes
    vgs = cfg["coupling"]["vic_grid_shape"]
    if not (isinstance(vgs, (list, tuple)) and len(vgs) == 2):
        raise ConfigError("coupling.vic_grid_shape must be a 2-item list [nrows, ncols]")


def _normalize_paths(cfg: dict[str, Any]) -> None:
    # expanduser on all obvious path-like entries
    def _xp(value: Any) -> Any:
        if isinstance(value, str) and ("/" in value or value.startswith("~")):
            return os.path.expanduser(value)
        return value

    for sec, keys in (
        ("mf6", ("workspace", "dll")),
        (
            "vic",
            ("dir", "exe", "global_param", "outputs_dir", "exchange_dir", "params_file"),
        ),
        ("coupling", ("table_csv",)),
    ):
        for k in keys:
            if k in cfg[sec]:
                cfg[sec][k] = _xp(cfg[sec][k])
